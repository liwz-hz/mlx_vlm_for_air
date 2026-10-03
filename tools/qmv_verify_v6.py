# v6: fork verify kernel 的精确算术 + 向量化 x 暂存（uint4 加载 + bf16 位平移）
# 目标: bit-exact 输出不变, 持续带宽超过 fork 的 18-22 GB/s

import time

import mlx.core as mx
import mlx.nn as nn

HEADER = r"""
    using namespace metal;

    constant constexpr int SIMD_SIZE = 32;
    constant constexpr int PACK_FACTOR = 8;
    constant constexpr int BYTES_PER_PACK = 4;
    constant constexpr int PACKS_PER_THREAD = 2;
    constant constexpr int VALUES_PER_THREAD = PACK_FACTOR * PACKS_PER_THREAD;
    constant constexpr int BLOCK_SIZE = VALUES_PER_THREAD * SIMD_SIZE;
    constant constexpr int GS = 64;
    constant constexpr int RESULTS_PER_SIMDGROUP = __RPS__;
    constant constexpr int NUM_SIMDGROUPS = 2;
    constant constexpr int BN = RESULTS_PER_SIMDGROUP * NUM_SIMDGROUPS;

    inline float bf16_hi(uint u) {
      return as_type<float>(u & 0xFFFF0000u);
    }
    inline float bf16_lo(uint u) {
      return as_type<float>(u << 16);
    }

    template <typename T>
    inline float load_vector_exact(const device T* x, thread float* x_thread) {
      float sum = 0.0f;
      const device uint4* xv = (const device uint4*)x;
      #pragma unroll
      for (int j = 0; j < VALUES_PER_THREAD / 8; j++) {
        uint4 w4 = xv[j];
        float v[8] = {
            bf16_lo(w4.x), bf16_hi(w4.x),
            bf16_lo(w4.y), bf16_hi(w4.y),
            bf16_lo(w4.z), bf16_hi(w4.z),
            bf16_lo(w4.w), bf16_hi(w4.w)};
        #pragma unroll
        for (int q = 0; q < 2; q++) {
          sum += float(T(v[4 * q]) + T(v[4 * q + 1]) + T(v[4 * q + 2]) + T(v[4 * q + 3]));
          x_thread[8 * j + 4 * q] = v[4 * q];
          x_thread[8 * j + 4 * q + 1] = v[4 * q + 1];
          x_thread[8 * j + 4 * q + 2] = v[4 * q + 2];
          x_thread[8 * j + 4 * q + 3] = v[4 * q + 3];
        }
      }
      return sum;
    }

    inline float qdot_exact(
        const device uint8_t* w,
        const thread float* x_thread,
        float scale,
        float bias,
        float sum) {
      float accum = 0.0f;
      const device uint16_t* ws = (const device uint16_t*)w;
      #pragma unroll
      for (int i = 0; i < (VALUES_PER_THREAD / 4); i++) {
        uint packed = ws[i];
        accum +=
            (x_thread[4 * i] * (packed & 0x000f) +
             x_thread[4 * i + 1] * ((packed >> 4) & 0x000f) +
             x_thread[4 * i + 2] * ((packed >> 8) & 0x000f) +
             x_thread[4 * i + 3] * ((packed >> 12) & 0x000f));
      }
      return scale * accum + sum * bias;
    }
""".replace("__RPS__", "__RPS_VALUE__")

SOURCE = r"""
    uint n_tile = threadgroup_position_in_grid.y;
    uint b_idx = threadgroup_position_in_grid.z;
    uint simd_gid = simdgroup_index_in_threadgroup;
    uint simd_lid = thread_index_in_simdgroup;

    int out_row = int(n_tile) * BN + int(simd_gid) * RESULTS_PER_SIMDGROUP;
    int in_vec_size_w = K_SIZE * BYTES_PER_PACK / PACK_FACTOR;
    int in_vec_size_g = K_SIZE / GS;

    const device uint8_t* ws_base =
        (const device uint8_t*)w + out_row * in_vec_size_w +
        int(simd_lid) * PACKS_PER_THREAD * BYTES_PER_PACK;
    const device T* scales_base =
        scales + out_row * in_vec_size_g + int(simd_lid) / (GS / VALUES_PER_THREAD);
    const device T* biases_base =
        biases + out_row * in_vec_size_g + int(simd_lid) / (GS / VALUES_PER_THREAD);
    const device T* x_base =
        x + int(b_idx) * VERIFY_T * K_SIZE + int(simd_lid) * VALUES_PER_THREAD;

    float result[VERIFY_T][RESULTS_PER_SIMDGROUP];
    float x_thread[VERIFY_T][VALUES_PER_THREAD];
    for (int t = 0; t < VERIFY_T; ++t) {
      for (int row = 0; row < RESULTS_PER_SIMDGROUP; ++row) {
        result[t][row] = 0.0f;
      }
    }

    const device uint8_t* ws = ws_base;
    const device T* sc = scales_base;
    const device T* bs = biases_base;
    const device T* xk = x_base;

    for (int k = 0; k < K_SIZE; k += BLOCK_SIZE) {
      float sums[VERIFY_T];
      for (int t = 0; t < VERIFY_T; ++t) {
        sums[t] = load_vector_exact<T>(xk + t * K_SIZE, x_thread[t]);
      }
      for (int row = 0; row < RESULTS_PER_SIMDGROUP; ++row) {
        const device uint8_t* wl = ws + row * in_vec_size_w;
        const device T* sl = sc + row * in_vec_size_g;
        const device T* bl = bs + row * in_vec_size_g;
        float s = sl[0];
        float b = bl[0];
        for (int t = 0; t < VERIFY_T; ++t) {
          result[t][row] += qdot_exact(wl, x_thread[t], s, b, sums[t]);
        }
      }
      ws += BLOCK_SIZE * BYTES_PER_PACK / PACK_FACTOR;
      sc += BLOCK_SIZE / GS;
      bs += BLOCK_SIZE / GS;
      xk += BLOCK_SIZE;
    }

    for (int row = 0; row < RESULTS_PER_SIMDGROUP; ++row) {
      int n = out_row + row;
      for (int t = 0; t < VERIFY_T; ++t) {
        float r = simd_sum(result[t][row]);
        if (simd_lid == 0) {
          y[(int(b_idx) * VERIFY_T + t) * N_SIZE + n] = T(r);
        }
      }
    }
"""


def make_v6(rps=4, dtype=mx.bfloat16):
    header = HEADER.replace("__RPS_VALUE__", str(rps))
    return mx.fast.metal_kernel(
        name="qmv_verify_v6_rps%d" % rps,
        input_names=["x", "w", "scales", "biases"],
        output_names=["y"],
        header=header,
        source=SOURCE,
    )


def v6_call(kernel, x, wq, s, b, T, K, N, rps, dtype=mx.bfloat16):
    B = x.shape[0]
    bn = rps * 2
    out = kernel(
        inputs=[x, wq, s, b],
        template=[("T", dtype), ("VERIFY_T", int(T)), ("K_SIZE", int(K)), ("N_SIZE", int(N))],
        grid=(32, 2 * (N // bn), B),
        threadgroup=(32, 2, 1),
        output_shapes=[(B, T, N)],
        output_dtypes=[dtype],
    )
    return out[0]


def make_ql(K, N, dtype=mx.bfloat16):
    ql = nn.QuantizedLinear.__new__(nn.QuantizedLinear)
    nn.Module.__init__(ql)
    ql.group_size, ql.bits, ql.mode = 64, 4, "affine"
    w = mx.random.normal((N, K)).astype(mx.float16)
    wq, s, b = mx.quantize(w, group_size=64, bits=4)
    ql.weight = wq
    ql.scales = s.astype(dtype)
    ql.biases = b.astype(dtype)
    ql.freeze()
    return ql


def bench(T, K, N, dtype=mx.bfloat16, reps=200):
    ql = make_ql(K, N, dtype)
    x = mx.contiguous(mx.random.normal((1, T, K)).astype(dtype))
    nbytes = ql.weight.nbytes + ql.scales.nbytes + ql.biases.nbytes
    from mlx_vlm.models.quantized_verifier import optimized_affine_linear
    ref_fork = optimized_affine_linear(ql, x)
    ref_seq = mx.concatenate(
        [ql(x[:, t : t + 1]) for t in range(T)], axis=1
    )
    for rps in (2, 4):
        k6 = make_v6(rps, dtype)
        got = v6_call(k6, x, ql.weight, ql.scales, ql.biases, T, K, N, rps, dtype)
        rel_qmm = (mx.abs(got.astype(mx.float32) - ref_fork.astype(mx.float32)).max() / mx.abs(ref_fork).max()).item()
        rel_seq = (mx.abs(got.astype(mx.float32) - ref_seq.astype(mx.float32)).max() / mx.abs(ref_seq).max()).item()
        def mine():
            y = v6_call(k6, x, ql.weight, ql.scales, ql.biases, T, K, N, rps, dtype); mx.eval(y)
        def fork():
            y = optimized_affine_linear(ql, x); mx.eval(y)
        def timed(f):
            for _ in range(20): f()
            t0 = time.perf_counter()
            for _ in range(reps): f()
            return (time.perf_counter() - t0) / reps * 1000
        t6, tf = timed(mine), timed(fork)
        print("T=%d (%5d->%5d) rps=%d: v6 %.3f ms (%.1f GB/s) vs fork %.3f ms (%.1f GB/s) | %.2fx | rel_qmm=%.5f rel_seq=%.5f" %
              (T, K, N, rps, t6, nbytes / t6 / 1e6, tf, nbytes / tf / 1e6, tf / t6, rel_qmm, rel_seq))


if __name__ == "__main__":
    bench(5, 5120, 17408)
    bench(5, 17408, 5120)
    bench(5, 5120, 12288)
    bench(5, 5120, 248320)
