# GEMV/GEMM bandwidth microbenchmarks for skinny-M quantized paths (see docs/local-serve-guide.md §6)
import time

import mlx.core as mx


def sustained_gemv(seconds=3.0):
    w = mx.random.normal((17408, 5120)).astype(mx.float16)
    x = mx.random.normal((1, 5120)).astype(mx.float16)
    chunks = []
    t_end = time.perf_counter() + seconds
    while time.perf_counter() < t_end:
        t0 = time.perf_counter()
        n = 0
        while time.perf_counter() - t0 < 0.5:
            y = x @ w.T
            mx.eval(y)
            n += 1
        dt = time.perf_counter() - t0
        nbytes = w.nbytes * n
        chunks.append(nbytes / dt / 1e9)
    return chunks


def big_gemm():
    a = mx.random.normal((2048, 2048)).astype(mx.float16)
    b = mx.random.normal((2048, 2048)).astype(mx.float16)
    for _ in range(3):
        c = a @ b
        mx.eval(c)
    t0 = time.perf_counter()
    n = 20
    for _ in range(n):
        c = a @ b
        mx.eval(c)
    dt = (time.perf_counter() - t0) / n
    tflops = 2 * 2048**3 / dt / 1e12
    return tflops


print("bf16 GEMV 持续 3s, 每 0.5s 的 GB/s:")
for i, gbs in enumerate(sustained_gemv(3.0)):
    print("  第%d个0.5s: %.1f GB/s" % (i + 1, gbs))
print("\n大 GEMM (2048^3 bf16): %.2f TFLOPS" % big_gemm())
