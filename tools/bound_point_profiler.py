"""Per-op prefill bound-point profiler for Qwen3.5-27B (M5 Air).

Measures achieved TFLOPS of every GEMM shape in the model at various M,
plus the delta-rule core, full-attention sdpa, and CPU sgemm for share
planning. Uses synthetic quantized weights (no model load).

Run: PYTHONPATH=. python3 tools/bound_point_profiler.py
"""
import time
import mlx.core as mx
import numpy as np

H = 5120          # hidden
I = 17408         # intermediate
KV = 10240        # in_proj_qkv out (2*key_dim + value_dim)
V = 6144          # value_dim (= q_proj out, in_proj_z out)
KDIM = 2048       # key_dim
VOCAB = 248320
KH, VH, HD = 16, 48, 128   # delta heads
AQH, AKVH, AHD = 24, 4, 256  # full-attn heads

G, B_ = 64, 4

def make_qw(K, N, seed=0):
    rng = np.random.default_rng(seed)
    w = mx.array(rng.standard_normal((N, K)).astype(np.float32) * 0.02).astype(mx.bfloat16)
    wq, s, b = mx.quantize(w, group_size=G, bits=B_)
    mx.eval(wq, s, b)
    return wq, s, b

def bench(fn, reps=5, warmup=2):
    for _ in range(warmup):
        fn()
    ts = []
    for _ in range(reps):
        t0 = time.perf_counter()
        fn()
        ts.append(time.perf_counter() - t0)
    return min(ts)

def bench_qmm(K, N, M, label):
    wq, s, b = make_qw(K, N)
    x = mx.array(np.random.default_rng(1).standard_normal((M, K)).astype(np.float32) * 0.1).astype(mx.bfloat16)
    def run():
        out = mx.quantized_matmul(x, wq, s, b, transpose=True, group_size=G, bits=B_)
        mx.eval(out)
    t = bench(run)
    tf = 2 * M * K * N / t / 1e12
    gb = (N * K * 0.5 + M * K * 2 + M * N * 2) / t / 1e9
    print(f"  {label:34s} K={K:6d} N={N:6d} M={M:5d}: {t*1e3:8.2f} ms  {tf:5.2f} TFLOPS  {gb:6.1f} GB/s")
    del wq, s, b, x
    return t, tf

def bench_bf16(K, N, M, label):
    rng = np.random.default_rng(2)
    w = mx.array(rng.standard_normal((N, K)).astype(np.float32) * 0.02).astype(mx.bfloat16)
    x = mx.array(rng.standard_normal((M, K)).astype(np.float32) * 0.1).astype(mx.bfloat16)
    def run():
        out = x @ w.T
        mx.eval(out)
    t = bench(run)
    tf = 2 * M * K * N / t / 1e12
    print(f"  {label:34s} K={K:6d} N={N:6d} M={M:5d}: {t*1e3:8.2f} ms  {tf:5.2f} TFLOPS")
    del w, x
    return t

def bench_dequant_mm(K, N, M, label):
    wq, s, b = make_qw(K, N)
    x = mx.array(np.random.default_rng(1).standard_normal((M, K)).astype(np.float32) * 0.1).astype(mx.bfloat16)
    def run():
        w = mx.dequantize(wq, s, b, group_size=G, bits=B_)
        out = x @ w.T
        mx.eval(out)
    t = bench(run)
    tf = 2 * M * K * N / t / 1e12
    print(f"  {label:34s} K={K:6d} N={N:6d} M={M:5d}: {t*1e3:8.2f} ms  {tf:5.2f} TFLOPS")
    del wq, s, b, x
    return t

def bench_delta_core(S):
    from mlx_vlm.models.qwen3_5.gated_delta import gated_delta_chunked
    rng = np.random.default_rng(3)
    q = mx.array(rng.standard_normal((1, S, KH, HD)).astype(np.float32) * 0.1).astype(mx.bfloat16)
    k = mx.array(rng.standard_normal((1, S, KH, HD)).astype(np.float32) * 0.1).astype(mx.bfloat16)
    v = mx.array(rng.standard_normal((1, S, VH, HD)).astype(np.float32) * 0.1).astype(mx.bfloat16)
    g = mx.zeros((1, S, VH)); beta = mx.ones((1, S, VH))
    def run():
        out = gated_delta_chunked(q, k, v, g, beta, state=None, C=64)
        mx.eval(out[0] if isinstance(out, tuple) else out)
    t = bench(run, reps=3)
    flops = S * (4 * 2 * KH * HD * HD + 4 * 2 * VH * HD * HD) * 1.0
    print(f"  delta_core S={S:5d}: {t*1e3:8.2f} ms  ~{flops/t/1e12:5.2f} TFLOPS (approx FLOPs)")
    return t

def bench_sdpa(S):
    rng = np.random.default_rng(4)
    q = mx.array(rng.standard_normal((1, AQH, S, AHD)).astype(np.float32) * 0.1).astype(mx.bfloat16)
    k = mx.array(rng.standard_normal((1, AKVH, S, AHD)).astype(np.float32) * 0.1).astype(mx.bfloat16)
    v = mx.array(rng.standard_normal((1, AKVH, S, AHD)).astype(np.float32) * 0.1).astype(mx.bfloat16)
    mask = "causal"
    def run():
        out = mx.fast.scaled_dot_product_attention(q, k, v, scale=AHD**-0.5, mask=mask)
        mx.eval(out)
    t = bench(run, reps=3)
    tf = 2 * 2 * AQH * S * S * AHD / t / 1e12
    print(f"  sdpa causal S={S:5d}: {t*1e3:8.2f} ms  {tf:5.2f} TFLOPS (dense)")
    return t

def bench_cpu(K, N, M):
    rng = np.random.default_rng(5)
    w = rng.standard_normal((N, K)).astype(np.float32) * 0.02
    x = rng.standard_normal((M, K)).astype(np.float32) * 0.1
    def run():
        out = x @ w.T
        return out
    t = bench(run, reps=3)
    tf = 2 * M * K * N / t / 1e12
    print(f"  CPU sgemm K={K:6d} N={N:6d} M={M:5d}: {t*1e3:8.2f} ms  {tf:5.2f} TFLOPS")
    return t

def bench_eval_overhead():
    a = mx.zeros((8, 8))
    mx.eval(a)
    def run():
        b = a + 1
        mx.eval(b)
    t = bench(run, reps=20)
    print(f"  mx.eval tiny-op overhead: {t*1e6:.0f} us")

if __name__ == "__main__":
    print("=== GPU qmm throughput (group=64 bits=4) ===")
    shapes = [
        (H, I, "MLP gate/up"),
        (I, H, "MLP down"),
        (H, KV, "in_proj_qkv"),
        (H, V, "in_proj_z"),
        (V, H, "out_proj"),
        (H, VOCAB, "lm_head"),
    ]
    for M in (512, 1024, 2048, 4096):
        print(f" -- M={M}")
        for K, N, label in shapes:
            bench_qmm(K, N, M, label)

    print("=== bf16 reference (peak compute) ===")
    for M in (2048, 4096):
        bench_bf16(H, I, M, "bf16 mm (MLP shape)")

    print("=== dequant + bf16 mm vs direct qmm (M=4096) ===")
    bench_dequant_mm(H, I, 4096, "dequant+mm MLP")
    bench_qmm(H, I, 4096, "qmm MLP (ref)")

    print("=== delta core + sdpa ===")
    for S in (2048, 4096):
        bench_delta_core(S)
        bench_sdpa(S)

    print("=== CPU (Accelerate sgemm, default threads) ===")
    for M in (256, 512, 1024):
        bench_cpu(H, I, M)
    bench_cpu(H, KV, 512)

    print("=== sync overhead ===")
    bench_eval_overhead()
