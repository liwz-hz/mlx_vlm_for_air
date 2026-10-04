# v7: fork verify kernel + per-row nibble unpack hoisting (bit-exact ALU cut)
# fork 的 qdot_exact 在每个 (row,t) 重复解包同一行权重; v7 每行解包一次跨 t 复用,
# 其余算术(含 bf16 逐步舍入的 x 和)与 fork 逐操作一致 -> 位精确。

import time

import mlx.core as mx
import mlx.nn as nn

from mlx_vlm.models.quantized_verifier import (
    _target_verify_qlinear_header,
    _TARGET_VERIFY_QMV_SOURCE,
    optimized_affine_linear,
)

V7_SOURCE = r"""
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
        scales + out_row * in_vec_size_g + int(simd_lid) / SCALE_STEP_PER_THREAD;
    const device T* biases_base =
        biases + out_row * in_vec_size_g + int(simd_lid) / SCALE_STEP_PER_THREAD;
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

      float n_f[VALUES_PER_THREAD];
      for (int row = 0; row < RESULTS_PER_SIMDGROUP; ++row) {
        const device uint8_t* wl = ws + row * in_vec_size_w;
        const device T* sl = sc + row * in_vec_size_g;
        const device T* bl = bs + row * in_vec_size_g;
        float s = float(sl[0]);
        float b = float(bl[0]);
        {
          const device uint16_t* ws16 = (const device uint16_t*)wl;
          #pragma unroll
          for (int i = 0; i < (VALUES_PER_THREAD / 4); i++) {
            uint packed = ws16[i];
            n_f[4 * i] = (packed & 0x000f);
            n_f[4 * i + 1] = ((packed >> 4) & 0x000f);
            n_f[4 * i + 2] = ((packed >> 8) & 0x000f);
            n_f[4 * i + 3] = ((packed >> 12) & 0x000f);
          }
        }
        for (int t = 0; t < VERIFY_T; ++t) {
          float accum = 0.0f;
          #pragma unroll
          for (int i = 0; i < (VALUES_PER_THREAD / 4); i++) {
            accum += (x_thread[t][4 * i] * n_f[4 * i] +
                      x_thread[t][4 * i + 1] * n_f[4 * i + 1] +
                      x_thread[t][4 * i + 2] * n_f[4 * i + 2] +
                      x_thread[t][4 * i + 3] * n_f[4 * i + 3]);
          }
          result[t][row] += s * accum + sums[t] * b;
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


def make_v7(T, K, N, dtype=mx.bfloat16):
    header = _target_verify_qlinear_header(4, 64, 4)
    return mx.fast.metal_kernel(
        name="verify_v7_t%d_k%d_n%d" % (T, K, N),
        input_names=["x", "w", "scales", "biases"],
        output_names=["y"],
        header=header,
        source=V7_SOURCE,
    )


def v7_call(kernel, x, wq, s, b, T, K, N, dtype=mx.bfloat16):
    B = x.shape[0]
    return kernel(
        inputs=[x, wq, s, b],
        template=[("T", dtype), ("VERIFY_T", int(T)), ("K_SIZE", int(K)), ("N_SIZE", int(N))],
        grid=(32, 2 * (N // 8), B),
        threadgroup=(32, 2, 1),
        output_shapes=[(B, T, N)],
        output_dtypes=[dtype],
    )[0]


def make_ql(K, N):
    ql = nn.QuantizedLinear.__new__(nn.QuantizedLinear)
    nn.Module.__init__(ql)
    ql.group_size, ql.bits, ql.mode = 64, 4, "affine"
    w = mx.random.normal((N, K)).astype(mx.float16)
    wq, s, b = mx.quantize(w, group_size=64, bits=4)
    ql.weight = wq
    ql.scales = s.astype(mx.bfloat16)
    ql.biases = b.astype(mx.bfloat16)
    ql.freeze()
    return ql


if __name__ == "__main__":
    print("== v7 位精确性 + 持续态性能 ==")
    for K, N in [(5120, 17408), (17408, 5120), (5120, 12288), (6144, 5120), (5120, 248320)]:
        ql = make_ql(K, N)
        x = mx.contiguous(mx.random.normal((1, 4, K)).astype(mx.bfloat16))
        nbytes = ql.weight.nbytes + ql.scales.nbytes + ql.biases.nbytes
        ref = optimized_affine_linear(ql, x)
        k7 = make_v7(4, K, N)
        got = v7_call(k7, x, ql.weight, ql.scales, ql.biases, 4, K, N)
        exact = bool(mx.array_equal(got, ref))

        def f7():
            y = v7_call(k7, x, ql.weight, ql.scales, ql.biases, 4, K, N)
            mx.eval(y)

        def ff():
            y = optimized_affine_linear(ql, x)
            mx.eval(y)

        def timed(f):
            for _ in range(20):
                f()
            best = 1e9
            for _ in range(3):
                t0 = time.perf_counter()
                for _ in range(100):
                    f()
                best = min(best, (time.perf_counter() - t0) / 100 * 1000)
            return best

        t7, tf = timed(f7), timed(ff)
        print("(%5d->%6d): v7 %.3f ms (%.1f GB/s) vs fork %.3f ms (%.1f GB/s) | %.2fx | bit-exact=%s" %
              (K, N, t7, nbytes / t7 / 1e6, tf, nbytes / tf / 1e6, tf / t7, exact))
