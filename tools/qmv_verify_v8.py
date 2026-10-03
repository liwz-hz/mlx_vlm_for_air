# v8: verify kernel with half2 paired FMA — 完全标量化 (无任何 thread 数组)
# 所有 t/j/q/i 循环在 Python 侧展开成字面量; Metal 侧只剩 k-loop 和 row-loop
# 精度: fp16 乘积 -> 偶发 1 bf16 ULP

import time

import mlx.core as mx
import mlx.nn as nn

from mlx_vlm.models.quantized_verifier import optimized_affine_linear

V8_HEADER = r"""
    using namespace metal;

    constant constexpr int SIMD_SIZE = 32;
    constant constexpr int PACK_FACTOR = 8;
    constant constexpr int BYTES_PER_PACK = 4;
    constant constexpr int PACKS_PER_THREAD = 2;
    constant constexpr int VALUES_PER_THREAD = PACK_FACTOR * PACKS_PER_THREAD;
    constant constexpr int BLOCK_SIZE = VALUES_PER_THREAD * SIMD_SIZE;
    constant constexpr int GS = 64;
    constant constexpr int RESULTS_PER_SIMDGROUP = 4;
    constant constexpr int NUM_SIMDGROUPS = 2;
    constant constexpr int BN = RESULTS_PER_SIMDGROUP * NUM_SIMDGROUPS;

    inline float bf16_hi_f(uint u) {
      return as_type<float>(u & 0xFFFF0000u);
    }
    inline float bf16_lo_f(uint u) {
      return as_type<float>(u << 16);
    }
"""

VPT = 16
NPAIR = VPT // 2  # half2 数量


def v8_source(verify_t):
    L = []
    A = L.append
    A(r"""
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
""")
    for t in range(verify_t):
        A("    half2 " + ", ".join("x%d_%d" % (t, i) for i in range(NPAIR)) + ";\n")
    A(r"""
    for (int rr = 0; rr < RESULTS_PER_SIMDGROUP; ++rr) {
""")
    for t in range(verify_t):
        A("      result[%d][rr] = 0.0f;\n" % t)
    A(r"""
    }

    const device uint8_t* ws = ws_base;
    const device T* sc = scales_base;
    const device T* bs = biases_base;
    const device T* xk = x_base;

    for (int k = 0; k < K_SIZE; k += BLOCK_SIZE) {
      float sums[VERIFY_T];
""")
    for t in range(verify_t):
        A("      {\n")
        A("        float sum = 0.0f;\n")
        A("        const device uint4* xv = (const device uint4*)(xk + %d * K_SIZE);\n" % t)
        for j in range(VPT // 8):
            A("        {\n")
            A("          uint4 w4 = xv[%d];\n" % j)
            A("          float v0 = bf16_lo_f(w4.x); float v1 = bf16_hi_f(w4.x);\n")
            A("          float v2 = bf16_lo_f(w4.y); float v3 = bf16_hi_f(w4.y);\n")
            A("          float v4 = bf16_lo_f(w4.z); float v5 = bf16_hi_f(w4.z);\n")
            A("          float v6 = bf16_lo_f(w4.w); float v7 = bf16_hi_f(w4.w);\n")
            A("          sum += (v0 + v1 + v2 + v3) + (v4 + v5 + v6 + v7);\n")
            A("          x%d_%d = half2(half(v0), half(v1));\n" % (t, 4 * j + 0))
            A("          x%d_%d = half2(half(v2), half(v3));\n" % (t, 4 * j + 1))
            A("          x%d_%d = half2(half(v4), half(v5));\n" % (t, 4 * j + 2))
            A("          x%d_%d = half2(half(v6), half(v7));\n" % (t, 4 * j + 3))
            A("        }\n")
        A("        sums[%d] = sum;\n" % t)
        A("      }\n")
    A(r"""
      for (int row = 0; row < RESULTS_PER_SIMDGROUP; ++row) {
        const device uint8_t* wl = ws + row * in_vec_size_w;
        const device T* sl = sc + row * in_vec_size_g;
        const device T* bl = bs + row * in_vec_size_g;
        float s = float(sl[0]);
        float b = float(bl[0]);
        {
          const device uint16_t* ws16 = (const device uint16_t*)wl;
""")
    for i in range(NPAIR):
        half_idx = i // 2
        sh0 = 4 * ((2 * i) % 4)
        A("          uint pk%d = ws16[%d];\n" % (i, half_idx))
        A("          n%d = half2(half((pk%d >> %d) & 0x000f), half((pk%d >> %d) & 0x000f));\n" % (i, i, sh0, i, sh0 + 4))
    A("        }\n")
    for t in range(verify_t):
        A("        {\n")
        A("          half2 acc = half2(0.0h, 0.0h);\n")
        for i in range(NPAIR):
            A("          acc = fma(n%d, x%d_%d, acc);\n" % (i, t, i))
        A("          float accum = float(acc.x) + float(acc.y);\n")
        A("          result[%d][row] += s * accum + sums[%d] * b;\n" % (t, t))
        A("        }\n")
    A(r"""
      }
      ws += BLOCK_SIZE * BYTES_PER_PACK / PACK_FACTOR;
      sc += BLOCK_SIZE / GS;
      bs += BLOCK_SIZE / GS;
      xk += BLOCK_SIZE;
    }

    for (int row = 0; row < RESULTS_PER_SIMDGROUP; ++row) {
      int n = out_row + row;
""")
    for t in range(verify_t):
        A("      {\n")
        A("        float r = simd_sum(result[%d][row]);\n" % t)
        A("        if (simd_lid == 0) {\n")
        A("          y[(int(b_idx) * VERIFY_T + %d) * N_SIZE + n] = T(r);\n" % t)
        A("        }\n")
        A("      }\n")
    A("    }\n")
    header_extra = "    half2 " + ", ".join("n%d" % i for i in range(NPAIR)) + ";\n"
    src = "".join(L)
    src = src.replace(
        "    const device uint8_t* ws = ws_base;",
        header_extra + "    const device uint8_t* ws = ws_base;",
    )
    return src


def make_v8(T, K, N, dtype=mx.bfloat16):
    return mx.fast.metal_kernel(
        name="verify_v8z_t%d_k%d_n%d" % (T, K, N),
        input_names=["x", "w", "scales", "biases"],
        output_names=["y"],
        header=V8_HEADER,
        source=v8_source(T),
    )


def v8_call(kernel, x, wq, s, b, T, K, N, dtype=mx.bfloat16):
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
    print("== v8z 完全标量版 ==")
    for K, N in [(512, 64), (5120, 17408), (17408, 5120), (5120, 12288), (6144, 5120), (5120, 248320)]:
        ql = make_ql(K, N)
        x = mx.contiguous(mx.random.normal((1, 4, K)).astype(mx.bfloat16))
        nbytes = ql.weight.nbytes + ql.scales.nbytes + ql.biases.nbytes
        ref = optimized_affine_linear(ql, x)
        k8 = make_v8(4, K, N)
        got = v8_call(k8, x, ql.weight, ql.scales, ql.biases, 4, K, N)
        rel = (mx.abs(got.astype(mx.float32) - ref.astype(mx.float32)).max() / mx.abs(ref).max()).item()

        def f8():
            y = v8_call(k8, x, ql.weight, ql.scales, ql.biases, 4, K, N)
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

        t8, tf = timed(f8), timed(ff)
        print("(%5d->%6d): v8z %.3f ms (%.1f GB/s) vs fork %.3f ms (%.1f GB/s) | %.2fx | rel=%.5f" %
              (K, N, t8, nbytes / t8 / 1e6, tf, nbytes / tf / 1e6, tf / t8, rel))
