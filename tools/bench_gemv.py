# GEMV/GEMM bandwidth microbenchmarks for skinny-M quantized paths (see docs/local-serve-guide.md §6)
import time

import mlx.core as mx


def bench_qmv(M, K, N, group=64, reps=300):
    w = mx.random.normal((N, K)).astype(mx.float16)
    x = mx.random.normal((M, K)).astype(mx.float16)
    wq, scales, biases = mx.quantize(w, group_size=group, bits=4)
    for _ in range(20):
        y = mx.quantized_matmul(x, wq, scales, biases, transpose=True, group_size=group, bits=4)
    mx.eval(y)
    t0 = time.perf_counter()
    for _ in range(reps):
        y = mx.quantized_matmul(x, wq, scales, biases, transpose=True, group_size=group, bits=4)
        mx.eval(y)
    dt = (time.perf_counter() - t0) / reps
    nbytes = wq.nbytes + scales.nbytes + biases.nbytes + x.nbytes
    return dt * 1000, nbytes / dt / 1e9


def bench_bf16_gemm(M, K, N, reps=100):
    w = mx.random.normal((N, K)).astype(mx.float16)
    x = mx.random.normal((M, K)).astype(mx.float16)
    for _ in range(10):
        y = x @ w.T
    mx.eval(y)
    t0 = time.perf_counter()
    for _ in range(reps):
        y = x @ w.T
        mx.eval(y)
    dt = (time.perf_counter() - t0) / reps
    nbytes = w.nbytes + x.nbytes
    return dt * 1000, nbytes / dt / 1e9


print("shape(M,K,N)          4bit-GEMV: ms   GB/s     | bf16: GB/s")
for M, K, N, label in [
    (1, 5120, 17408, "gate/up (MLP, batch=1)"),
    (1, 5120, 34816, "gate+up 融合 (batch=1)"),
    (4, 5120, 17408, "gate/up (batch=4)"),
    (8, 5120, 17408, "gate/up (batch=8)"),
    (1, 17408, 5120, "down (batch=1)"),
    (1, 5120, 248320, "lm_head (batch=1)"),
    (64, 5120, 17408, "gate/up (batch=64)"),
]:
    ms4, gbs4 = bench_qmv(M, K, N)
    msb, gbsb = bench_bf16_gemm(M, K, N)
    print(
        f"({M:>3},{K},{N:>6}) {label:<24} {ms4:7.3f} {gbs4:6.1f}     | {gbsb:6.1f}"
    )
