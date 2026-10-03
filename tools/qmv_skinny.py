# Skinny-M (M<=8) affine 4-bit GEMV custom kernel via mx.fast.metal_kernel.
# Targets the qmv_wide underperformance measured in tools/bench_gemv.py.

import time

import mlx.core as mx

SOURCE = """
    const int row = threadgroup_position_in_grid.x;
    const int lane = thread_position_in_threadgroup.x % 32;
    const int sg = simdgroup_index_in_threadgroup;
    const int n_groups = K / GROUP;
    const int stride = 32 * NSG;

    float acc[M];
    for (int m = 0; m < M; m++) acc[m] = 0.0f;

    const device uint* wrow = w + (size_t)row * (K / 8);
    const device half* srow = s + (size_t)row * n_groups;
    const device half* brow = b + (size_t)row * n_groups;

    for (int g = sg * 32 + lane; g < n_groups; g += stride) {
        const float sc = float(srow[g]);
        const float bi = float(brow[g]);
        const device uint* wp = wrow + g * (GROUP / 8);
        for (int c = 0; c < GROUP / 8; c++) {
            uint packed = wp[c];
            int base = g * GROUP + c * 8;
            float wv[8];
            #pragma unroll
            for (int i = 0; i < 8; i++) {
                wv[i] = fma((float)((packed >> (4 * i)) & 0xF), sc, bi);
            }
            #pragma unroll
            for (int m = 0; m < M; m++) {
                uint4 xw = *((const device uint4*)(x + (size_t)m * K + base));
                half2 h0 = as_type<half2>(xw.x);
                half2 h1 = as_type<half2>(xw.y);
                half2 h2 = as_type<half2>(xw.z);
                half2 h3 = as_type<half2>(xw.w);
                acc[m] += wv[0] * float(h0.x) + wv[1] * float(h0.y)
                        + wv[2] * float(h1.x) + wv[3] * float(h1.y)
                        + wv[4] * float(h2.x) + wv[5] * float(h2.y)
                        + wv[6] * float(h3.x) + wv[7] * float(h3.y);
            }
        }
    }

    threadgroup float partials[NSG][M];
    for (int m = 0; m < M; m++) {
        float v = simd_sum(acc[m]);
        if (lane == 0) partials[sg][m] = v;
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    if (thread_position_in_threadgroup.x == 0) {
        for (int m = 0; m < M; m++) {
            float total = 0.0f;
            #pragma unroll
            for (int i = 0; i < NSG; i++) total += partials[i][m];
            out[(size_t)m * N + row] = half(total);
        }
    }
"""


SOURCE_V5 = """
    const int row = threadgroup_position_in_grid.x;
    const int lane = thread_position_in_threadgroup.x % 32;
    const int sg = simdgroup_index_in_threadgroup;
    const int n_groups = K / GROUP;
    const int stride = 32 * NSG;

    float acc[M];
    for (int m = 0; m < M; m++) acc[m] = 0.0f;

    const device uint* wrow = w + (size_t)row * (K / 8);
    const device half* srow = s + (size_t)row * n_groups;
    const device half* brow = b + (size_t)row * n_groups;

    for (int g = sg * 32 + lane; g < n_groups; g += stride) {
        const float sc = float(srow[g]);
        const float bi = float(brow[g]);
        const device uint* wp = wrow + g * (GROUP / 8);
        half2 accn[M];
        #pragma unroll
        for (int m = 0; m < M; m++) accn[m] = half2(0.0h, 0.0h);
        for (int c = 0; c < GROUP / 8; c++) {
            uint packed = wp[c];
            int base = g * GROUP + c * 8;
            half2 nh[4];
            #pragma unroll
            for (int i = 0; i < 4; i++) {
                uint byte = (packed >> (8 * i)) & 0xFF;
                nh[i] = half2(half(byte & 0xF), half((byte >> 4) & 0xF));
            }
            #pragma unroll
            for (int m = 0; m < M; m++) {
                uint4 xw = *((const device uint4*)(x + (size_t)m * K + base));
                half2 x0 = as_type<half2>(xw.x);
                half2 x1 = as_type<half2>(xw.y);
                half2 x2 = as_type<half2>(xw.z);
                half2 x3 = as_type<half2>(xw.w);
                accn[m] = fma(nh[0], x0, accn[m]);
                accn[m] = fma(nh[1], x1, accn[m]);
                accn[m] = fma(nh[2], x2, accn[m]);
                accn[m] = fma(nh[3], x3, accn[m]);
            }
        }
        #pragma unroll
        for (int m = 0; m < M; m++) {
            float dots = float(accn[m].x) + float(accn[m].y);
            acc[m] = fma(sc, dots, acc[m]);
            acc[m] = fma(bi, xsum[(size_t)m * n_groups + g], acc[m]);
        }
    }

    threadgroup float partials[NSG][M];
    for (int m = 0; m < M; m++) {
        float v = simd_sum(acc[m]);
        if (lane == 0) partials[sg][m] = v;
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    if (thread_position_in_threadgroup.x == 0) {
        for (int m = 0; m < M; m++) {
            float total = 0.0f;
            #pragma unroll
            for (int i = 0; i < NSG; i++) total += partials[i][m];
            out[(size_t)m * N + row] = half(total);
        }
    }
"""


def make_kernel():
    return mx.fast.metal_kernel(
        name="qmv_skinny",
        input_names=["x", "w", "s", "b", "K", "N"],
        output_names=["out"],
        source=SOURCE,
    )


def make_kernel_v5():
    return mx.fast.metal_kernel(
        name="qmv_skinny_v5",
        input_names=["x", "w", "s", "b", "xsum", "K", "N"],
        output_names=["out"],
        source=SOURCE_V5,
    )


def qmv_skinny_v5(x, wq, s, b, kernel, M, group=64, nsg=1):
    M_, K = x.shape
    N = wq.shape[0]
    n_groups = K // group
    xsum = mx.sum(x.reshape(M_, n_groups, group).astype(mx.float32), axis=-1)
    out = kernel(
        inputs=[x, wq, s, b, xsum, K, N],
        grid=(N * 32 * nsg, 1, 1),
        threadgroup=(32 * nsg, 1, 1),
        output_shapes=[(M_, N)],
        output_dtypes=[mx.float16],
        template=[("M", M), ("GROUP", group), ("NSG", nsg)],
    )
    return out[0]


def qmv_skinny(x, wq, s, b, kernel, M, group=64, nsg=1):
    M_, K = x.shape
    N = wq.shape[0]
    out = kernel(
        inputs=[x, wq, s, b, K, N],
        grid=(N * 32 * nsg, 1, 1),
        threadgroup=(32 * nsg, 1, 1),
        output_shapes=[(M_, N)],
        output_dtypes=[mx.float16],
        template=[("M", M), ("GROUP", group), ("NSG", nsg)],
    )
    return out[0]


def correctness():
    mx.random.seed(0)
    K, N, group = 512, 128, 64
    w = mx.random.normal((N, K)).astype(mx.float16)
    wq, s, b = mx.quantize(w, group_size=group, bits=4)
    kernel = make_kernel()
    for M in (1, 2, 4, 8):
        for nsg in (1, 2, 4):
            x = mx.random.normal((M, K)).astype(mx.float16)
            ref = mx.quantized_matmul(x, wq, s, b, transpose=True, group_size=group, bits=4)
            got = qmv_skinny(x, wq, s, b, kernel, M, group, nsg)
            err = mx.abs(got.astype(mx.float32) - ref.astype(mx.float32)).max().item()
            rel = err / mx.abs(ref).max().item()
            print("M=%d nsg=%d rel=%.5f %s" % (M, nsg, rel, "OK" if rel < 5e-3 else "FAIL"))


def bench(M, K=5120, N=17408, group=64, reps=200):
    w = mx.random.normal((N, K)).astype(mx.float16)
    wq, s, b = mx.quantize(w, group_size=group, bits=4)
    x = mx.random.normal((M, K)).astype(mx.float16)
    kernel = make_kernel()
    nbytes = wq.nbytes + s.nbytes + b.nbytes
    for nsg in (1, 2, 4):
        for _ in range(10):
            y = qmv_skinny(x, wq, s, b, kernel, M, group, nsg)
            mx.eval(y)
        t0 = time.perf_counter()
        for _ in range(reps):
            y = qmv_skinny(x, wq, s, b, kernel, M, group, nsg)
            mx.eval(y)
        t = (time.perf_counter() - t0) / reps * 1000
        print(
            "custom M=%d nsg=%d: %.3f ms  wall_bw %.1f GB/s  per-token %.3f ms"
            % (M, nsg, t, nbytes / t / 1e6, t / M)
        )


if __name__ == "__main__":
    print("== correctness ==")
    correctness()
    print("== perf (vs MLX per-token: M=4: 0.621ms, M=8: 0.516ms) ==")
    for M in (4, 8):
        bench(M)
