"""Verify MLX 4-bit group quantization packing layout for CPU-side unpacking,
and measure fp16 vs fp32 CPU GEMM rates (AMX)."""
import time
import mlx.core as mx
import numpy as np

G, BITS = 64, 4

# --- 1. Packing layout ---
N, K = 8, 128
rng = np.random.default_rng(0)
w_np = (rng.standard_normal((N, K)) * 0.5).astype(np.float32)
w = mx.array(w_np)
wq, s, b = mx.quantize(w, group_size=G, bits=BITS)
mx.eval(wq, s, b)
w_ref = np.array(mx.dequantize(wq, s, b, group_size=G, bits=BITS))

print("packed dtype:", wq.dtype, "shape:", wq.shape)   # expect [N, K*4/32] uint32
print("scales dtype:", s.dtype, "shape:", s.shape)      # [N, K/G]
print("biases dtype:", b.dtype, "shape:", b.shape)

wq_np = np.array(wq)
s_np = np.array(s)
b_np = np.array(b)

# Hypothesis: word j along packed axis holds 8 consecutive 4-bit values:
# element index e = word*8 + nibble, value = (word >> (4*nibble)) & 0xF
# group g = e // 64; scale/bias index g
def unpack_naive(wq_np, s_np, b_np, n, k):
    out = np.empty((n, k), dtype=np.float32)
    n_words = wq_np.shape[1]
    for r in range(n):
        for wi in range(n_words):
            word = int(wq_np[r, wi])
            for nb in range(8):
                e = wi * 8 + nb
                g = e // G
                q = (word >> (4 * nb)) & 0xF
                out[r, e] = q * float(s_np[r, g]) + float(b_np[r, g])
    return out

w_try = unpack_naive(wq_np, s_np, b_np, N, K)
print("naive unpack matches:", np.array_equal(w_try, w_ref))
if not np.array_equal(w_try, w_ref):
    # try reversed nibble order or transposed
    print("ref[0,:16]:", w_ref[0, :16])
    print("try[0,:16]:", w_try[0, :16])

# --- 2. Vectorized unpacker (candidate for production) ---
def unpack_vec(wq_np, s_np, b_np):
    n, n_words = wq_np.shape
    k = n_words * 8
    n_groups = k // G
    words_per_group = G // 8
    # nibbles: [n, n_words, 8]
    shifts = (np.arange(8, dtype=np.uint32) * 4)
    nib = ((wq_np[:, :, None] >> shifts[None, None, :]) & 0xF).astype(np.uint32)
    q = nib.reshape(n, k)  # element order = word*8 + nibble
    # group index per element
    g_idx = (np.arange(k) // G)
    sc = s_np[:, g_idx].astype(np.float32)
    bi = b_np[:, g_idx].astype(np.float32)
    return q.astype(np.float32) * sc + bi

w_vec = unpack_vec(wq_np, s_np, b_np)
print("vectorized unpack matches:", np.array_equal(w_vec, w_ref))

# --- 3. fp16 vs fp32 CPU GEMM rate ---
M_, K_, N_ = 2048, 5120, 2089
a32 = rng.standard_normal((M_, K_)).astype(np.float32) * 0.1
b32 = rng.standard_normal((N_, K_)).astype(np.float32) * 0.02

t0 = time.perf_counter(); c = a32 @ b32.T; t32 = time.perf_counter() - t0
print(f"fp32 sgemm: {t32*1e3:.1f} ms  {2*M_*K_*N_/t32/1e12:.2f} TFLOPS")

a16 = a32.astype(np.float16); b16 = b32.astype(np.float16)
t0 = time.perf_counter(); c16 = a16 @ b16.T; t16 = time.perf_counter() - t0
print(f"fp16 gemm : {t16*1e3:.1f} ms  {2*M_*K_*N_/t16/1e12:.2f} TFLOPS  maxdiff={np.abs(c16.astype(np.float32)-c).max():.4f}")

# repeat fp32 to confirm steady state
t0 = time.perf_counter(); c = a32 @ b32.T; t32b = time.perf_counter() - t0
t0 = time.perf_counter(); c16 = a16 @ b16.T; t16b = time.perf_counter() - t0
print(f"steady: fp32 {2*M_*K_*N_/t32b/1e12:.2f} TFLOPS   fp16 {2*M_*K_*N_/t16b/1e12:.2f} TFLOPS")
