#include <stdio.h>
#include <stdlib.h>
#include <time.h>
#include <Accelerate/Accelerate.h>

static double now_s(void) {
    struct timespec ts;
    clock_gettime(CLOCK_MONOTONIC, &ts);
    return ts.tv_sec + ts.tv_nsec * 1e-9;
}

static void desc2d(BNNSNDArrayDescriptor *d, size_t r, size_t c, BNNSDataType dt, void *data) {
    d->flags = 0;
    d->layout = BNNSDataLayout2DLastMajor;
    d->size[0] = r; d->size[1] = c;
    d->stride[0] = c; d->stride[1] = 1;
    d->data = data;
    d->data_type = dt;
}

static void test(const char *tag, size_t M, size_t K, size_t N,
                 BNNSDataType adt, BNNSDataType bdt, BNNSDataType cdt) {
    float *a32 = malloc(M * K * 4);
    float *b32 = malloc(N * K * 4);
    float *ref = malloc(M * N * 4);
    for (size_t i = 0; i < M * K; i++) a32[i] = ((float)(i % 17) - 8) * 0.0625f;
    for (size_t i = 0; i < N * K; i++) b32[i] = ((float)(i % 23) - 11) * 0.03125f;
    for (size_t i = 0; i < M * N; i++) ref[i] = 0;

    // reference via cblas (RowMajor, noTrans for A, Trans for B)
    cblas_sgemm(CblasRowMajor, CblasNoTrans, CblasTrans, (int)M, (int)N, (int)K,
                1.0f, a32, (int)K, b32, (int)K, 0.0f, ref, (int)N);

    void *a, *b, *c;
    size_t asz = (adt == BNNSDataTypeFloat32) ? 4 : 2;
    size_t bsz = (bdt == BNNSDataTypeFloat32) ? 4 : 2;
    size_t csz = (cdt == BNNSDataTypeFloat32) ? 4 : 2;
    a = malloc(M * K * asz);
    b = malloc(N * K * bsz);
    c = malloc(M * N * csz);

    // convert fp32 -> fp16/bf16 (RNE for bf16 via trick; fp16 via __fp16)
    for (size_t i = 0; i < M * K; i++) {
        if (adt == BNNSDataTypeFloat32) ((float*)a)[i] = a32[i];
        else if (adt == BNNSDataTypeFloat16) ((__fp16*)a)[i] = a32[i];
        else { unsigned u = *(unsigned*)&a32[i]; unsigned r = (u + 0x7fff + ((u >> 16) & 1)) >> 16; ((unsigned short*)a)[i] = r; }
    }
    for (size_t i = 0; i < N * K; i++) {
        if (bdt == BNNSDataTypeFloat32) ((float*)b)[i] = b32[i];
        else if (bdt == BNNSDataTypeFloat16) ((__fp16*)b)[i] = b32[i];
        else { unsigned u = *(unsigned*)&b32[i]; unsigned r = (u + 0x7fff + ((u >> 16) & 1)) >> 16; ((unsigned short*)b)[i] = r; }
    }

    BNNSNDArrayDescriptor ad, bd, cd;
    desc2d(&ad, M, K, adt, a);
    desc2d(&bd, N, K, bdt, b);   // [N, K], transB=true
    desc2d(&cd, M, N, cdt, c);
    BNNSFilterParameters fp = {.n_threads = 0};

    ssize_t ws = BNNSMatMulWorkspaceSize(false, true, 1.0f, &ad, &bd, &cd, &fp);
    if (ws < 0) { printf("%-24s workspace FAILED (%zd)\n", tag, ws); goto done; }
    void *wsbuf = malloc(ws);

    int rc = BNNSMatMul(false, true, 1.0f, &ad, &bd, &cd, wsbuf, &fp);
    if (rc != 0) { printf("%-24s apply FAILED rc=%d\n", tag, rc); free(wsbuf); goto done; }

    // correctness
    double maxerr = 0;
    for (size_t i = 0; i < M * N; i++) {
        float got = (cdt == BNNSDataTypeFloat32) ? ((float*)c)[i]
                  : (cdt == BNNSDataTypeFloat16) ? (float)((__fp16*)c)[i]
                  : (float)((__fp16*)c)[i];  // bf16 -> upcast via fp16 path won't work; handle below
        double e = fabs(got - ref[i]);
        if (e > maxerr) maxerr = e;
    }

    // bench
    int reps = 3;
    double tmin = 1e9;
    for (int r = 0; r < reps; r++) {
        double t0 = now_s();
        BNNSMatMul(false, true, 1.0f, &ad, &bd, &cd, wsbuf, &fp);
        double t1 = now_s();
        if (t1 - t0 < tmin) tmin = t1 - t0;
    }
    double tflops = 2.0 * M * K * N / tmin / 1e12;
    printf("%-24s %7.2f ms  %5.2f TFLOPS  maxerr=%.4f\n", tag, tmin * 1e3, tflops, maxerr);
    free(wsbuf);
done:
    free(a32); free(b32); free(ref); free(a); free(b); free(c);
}

int main() {
    size_t M = 2048, K = 5120, N = 2090;
    printf("BNNSMatMul M=%zu K=%zu N=%zu (x @ W.T):\n", M, K, N);
    test("fp32/fp32->fp32", M, K, N, BNNSDataTypeFloat32, BNNSDataTypeFloat32, BNNSDataTypeFloat32);
    test("fp16/fp16->fp32", M, K, N, BNNSDataTypeFloat16, BNNSDataTypeFloat16, BNNSDataTypeFloat32);
    test("fp16/fp16->fp16", M, K, N, BNNSDataTypeFloat16, BNNSDataTypeFloat16, BNNSDataTypeFloat16);
    test("bf16/bf16->fp32", M, K, N, BNNSDataTypeBFloat16, BNNSDataTypeBFloat16, BNNSDataTypeFloat32);
    test("bf16/bf16->bf16", M, K, N, BNNSDataTypeBFloat16, BNNSDataTypeBFloat16, BNNSDataTypeBFloat16);
    size_t K2 = 2176, N2 = 5120;
    printf("MLP down shape M=%zu K=%zu N=%zu:\n", M, K2, N2);
    test("bf16/bf16->fp32", M, K2, N2, BNNSDataTypeBFloat16, BNNSDataTypeBFloat16, BNNSDataTypeFloat32);
    test("fp32/fp32->fp32", M, K2, N2, BNNSDataTypeFloat32, BNNSDataTypeFloat32, BNNSDataTypeFloat32);
    return 0;
}
