"""Benchmark production gated_delta_kernel path vs chunked ops path at prefill T."""
import time
import mlx.core as mx
import numpy as np
from mlx_vlm.models.qwen3_5.gated_delta import (
    gated_delta_kernel,
    gated_delta_chunked,
    compute_g,
)

KH, VH, HD = 16, 48, 128

def bench(fn, reps=3, warmup=1):
    for _ in range(warmup):
        fn()
    ts = []
    for _ in range(reps):
        t0 = time.perf_counter()
        fn()
        ts.append(time.perf_counter() - t0)
    return min(ts)

for T in (1024, 2048, 4096):
    rng = np.random.default_rng(0)
    q = mx.array(rng.standard_normal((1, T, KH, HD)).astype(np.float32) * 0.1).astype(mx.bfloat16)
    k = mx.array(rng.standard_normal((1, T, KH, HD)).astype(np.float32) * 0.1).astype(mx.bfloat16)
    v = mx.array(rng.standard_normal((1, T, VH, HD)).astype(np.float32) * 0.1).astype(mx.bfloat16)
    a = mx.zeros((1, T, VH)); b = mx.ones((1, T, VH))
    A_log = mx.zeros(VH); dt_bias = mx.zeros(VH)
    state = mx.zeros((1, VH, HD, HD), mx.float32)

    g = compute_g(A_log, a, dt_bias)

    def run_kernel():
        out = gated_delta_kernel(q, k, v, g, mx.sigmoid(b) * 0.5, state, None)
        mx.eval(out)

    tk = bench(run_kernel)
    # bytes moved (ideal)
    bytes_ = (q.nbytes + k.nbytes + v.nbytes + 2 * 4096 * 0 + q.nbytes // KH * VH * 2) if False else (
        q.nbytes + k.nbytes + v.nbytes + (T * VH * HD * 2) * 2 + state.nbytes * 2
    )
    print(f"T={T:5d} kernel:  {tk*1e3:8.2f} ms  ({bytes_/tk/1e9:6.1f} GB/s effective)")

    for C in (64, 128, 256):
        def run_chunked():
            out = gated_delta_chunked(q, k, v, g, mx.sigmoid(b) * 0.5, state, C=C)
            mx.eval(out[0], out[1])
        try:
            tc = bench(run_chunked)
            print(f"T={T:5d} chunked C={C:4d}: {tc*1e3:8.2f} ms")
        except Exception as e:
            print(f"T={T:5d} chunked C={C:4d}: FAILED {e}")
