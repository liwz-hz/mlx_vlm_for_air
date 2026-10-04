/* Fused CPU-share kernels for prefill co-execution (single ctypes call
 * releases the GIL for the whole batch -> no contention with MLX's async
 * dispatch thread, which measured +40% CPU-share wall time otherwise).
 *
 * mlp_cpu_share:  gate/up via BNNSMatMul (bf16 in, fp32 out, AMX),
 *                 NEON swiglu in-place, down-proj K-slice via cblas_sgemm.
 * split_gemm_bf16: single bf16 GEMM (delta-net qkv/z row-split share).
 */

#include <Accelerate/Accelerate.h>
#include <arm_neon.h>
#include <stdint.h>
#include <stdlib.h>
#include <string.h>
#include <pthread/qos.h>

/* ---- fast sigmoid (same kernel as swiglu_neon.c) ---- */
static inline float32x4_t fast_sigmoid(float32x4_t g) {
    g = vminq_f32(vmaxq_f32(g, vdupq_n_f32(-15.0f)), vdupq_n_f32(15.0f));
    const float32x4_t LOG2E = vdupq_n_f32(1.4426950408889634f);
    float32x4_t z = vmulq_f32(vnegq_f32(g), LOG2E);
    float32x4_t fn = vrndmq_f32(z);
    float32x4_t f = vsubq_f32(z, fn);
    const float32x4_t c4 = vdupq_n_f32(0.00899777f);
    const float32x4_t c3 = vdupq_n_f32(0.0558282f);
    const float32x4_t c2 = vdupq_n_f32(0.240236f);
    const float32x4_t c1 = vdupq_n_f32(0.6933520f);
    const float32x4_t c0 = vdupq_n_f32(1.0f);
    float32x4_t p = vdupq_n_f32(0.00139721f);
    p = vfmaq_f32(c4, f, p);
    p = vfmaq_f32(c3, f, p);
    p = vfmaq_f32(c2, f, p);
    p = vfmaq_f32(c1, f, p);
    p = vfmaq_f32(c0, f, p);
    int32x4_t ni = vcvtq_s32_f32(fn);
    ni = vaddq_s32(ni, vdupq_n_s32(127));
    ni = vshlq_n_s32(ni, 23);
    float32x4_t scale = vreinterpretq_f32_s32(ni);
    float32x4_t e = vmulq_f32(p, scale);
    float32x4_t denom = vaddq_f32(vdupq_n_f32(1.0f), e);
    float32x4_t r = vrecpeq_f32(denom);
    r = vmulq_f32(vrecpsq_f32(denom, r), r);
    r = vmulq_f32(vrecpsq_f32(denom, r), r);
    return r;
}

void swiglu_neon(const float* gate, const float* up, float* out, int n) {
    int i = 0;
    for (; i + 3 < n; i += 4) {
        float32x4_t g = vld1q_f32(gate + i);
        float32x4_t u = vld1q_f32(up + i);
        float32x4_t sig = fast_sigmoid(g);
        float32x4_t silu = vmulq_f32(g, sig);
        vst1q_f32(out + i, vmulq_f32(silu, u));
    }
    for (; i < n; i++) {
        float g = gate[i];
        g = g > 15.0f ? 15.0f : (g < -15.0f ? -15.0f : g);
        float z = -g * 1.4426950408889634f;
        float fn_ = floorf(z);
        float fr = z - fn_;
        float p = 1.0f + fr * (0.6933520f + fr * (0.240236f + fr * (0.0558282f
                   + fr * (0.00899777f + fr * 0.00139721f))));
        union { int32_t i; float f; } s = { .i = (int32_t)(fn_ + 127) << 23 };
        float e = p * s.f;
        out[i] = g * (1.0f / (1.0f + e)) * up[i];
    }
}

/* ---- BNNS bf16 GEMM: C[M,N] fp32 = A[M,K] bf16 @ B[N,K] bf16 ^T ---- */
static int bnns_gemm(const unsigned short *a, const unsigned short *b,
                     float *c, size_t m, size_t k, size_t n) {
    pthread_set_qos_class_self_np(QOS_CLASS_USER_INTERACTIVE, 0);
    BNNSNDArrayDescriptor ad = {
        .layout = BNNSDataLayout2DLastMajor,
        .size = {m, k},
        .stride = {k, 1},
        .data = (void *)a,
        .data_type = BNNSDataTypeBFloat16};
    BNNSNDArrayDescriptor bd = {
        .layout = BNNSDataLayout2DLastMajor,
        .size = {n, k},
        .stride = {k, 1},
        .data = (void *)b,
        .data_type = BNNSDataTypeBFloat16};
    BNNSNDArrayDescriptor cd = {
        .layout = BNNSDataLayout2DLastMajor,
        .size = {m, n},
        .stride = {n, 1},
        .data = (void *)c,
        .data_type = BNNSDataTypeFloat32};
    BNNSFilterParameters fp = {.n_threads = 8};
    ssize_t ws = BNNSMatMulWorkspaceSize(false, true, 1.0f, &ad, &bd, &cd, &fp);
    if (ws < 0) return -1;
    void *wsbuf = ws > 0 ? malloc((size_t)ws) : NULL;
    int rc = BNNSMatMul(false, true, 1.0f, &ad, &bd, &cd, wsbuf, &fp);
    free(wsbuf);
    return rc;
}

void split_gemm_bf16(const unsigned short *x, const unsigned short *w,
                     float *out, size_t m, size_t k, size_t n) {
    bnns_gemm(x, w, out, m, k, n);
}

/* MLP CPU share.
 * x:   [M, K]        bf16 (view of MLX buffer)
 * wg:  [NC, K]       bf16 (dequantized gate rows)
 * wu:  [NC, K]       bf16 (dequantized up rows)
 * wd:  [N_DOWN, NC]  fp32 (dequantized down K-slice columns)
 * out: [M, N_DOWN]   fp32 (view of MLX buffer)
 */
int mlp_cpu_share(const unsigned short *x, const unsigned short *wg,
                  const unsigned short *wu, const float *wd, float *out,
                  size_t m, size_t k, size_t nc, size_t n_down) {
    float *gate = malloc(m * nc * sizeof(float));
    float *up = malloc(m * nc * sizeof(float));
    if (!gate || !up) {
        free(gate);
        free(up);
        return -1;
    }
    int rc = bnns_gemm(x, wg, gate, m, k, nc);
    if (rc == 0) rc = bnns_gemm(x, wu, up, m, k, nc);
    if (rc == 0) swiglu_neon(gate, up, gate, (int)(m * nc));
    if (rc == 0) {
        /* out[M, N_DOWN] = gate[M, NC] @ wd[N_DOWN, NC]^T */
        cblas_sgemm(CblasRowMajor, CblasNoTrans, CblasTrans,
                    (int)m, (int)n_down, (int)nc, 1.0f, gate, (int)nc,
                    wd, (int)nc, 0.0f, out, (int)n_down);
    }
    free(gate);
    free(up);
    return rc;
}
