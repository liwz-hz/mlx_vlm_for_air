"""Final unit test: MLP-only K-split co-execution — correctness + speed."""
import os
import sys
import time

os.environ["MLX_VLM_CPU_PREFILL"] = "0.11"

import mlx.core as mx
import mlx.nn as nn
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from mlx_vlm import cpu_prefill
from mlx_vlm.models.qwen3_5.language import Qwen3_5MLP

M = 2048


def make_qlinear(n_in, n_out, seed):
    w = mx.array(
        (np.random.default_rng(seed).standard_normal((n_out, n_in)) * 0.02).astype(np.float32)
    ).astype(mx.bfloat16)
    wq, s, b = mx.quantize(w, group_size=64, bits=4)
    lin = nn.QuantizedLinear(n_in, n_out, bias=False, group_size=64, bits=4)
    lin.weight, lin.scales, lin.biases = wq, s, b
    return lin


mlp = Qwen3_5MLP(5120, 17408)
mlp.gate_proj = make_qlinear(5120, 17408, 1)
mlp.down_proj = make_qlinear(17408, 5120, 2)
mlp.up_proj = make_qlinear(5120, 17408, 3)

x = mx.array(np.random.default_rng(9).standard_normal((1, M, 5120)) * 0.5).astype(mx.bfloat16)
mx.eval(x)

orig_call = Qwen3_5MLP.__call__
ref = mlp(x)
mx.eval(ref)


def bench(fn, reps=5):
    fn()
    ts = []
    for _ in range(reps):
        t0 = time.perf_counter()
        fn()
        ts.append(time.perf_counter() - t0)
    return min(ts)


Qwen3_5MLP.__call__ = cpu_prefill._hybrid_mlp_call
cpu_prefill._orig_mlp_call = orig_call

out = mlp(x)
mx.eval(out)
ref_np = np.array(ref.astype(mx.float32))
out_np = np.array(out.astype(mx.float32))
diff = np.abs(out_np - ref_np)
row_rms = np.sqrt((ref_np.reshape(-1, 5120) ** 2).mean(axis=1, keepdims=True))
rel = (diff.reshape(-1, 5120) / (row_rms + 1e-6))
print(f"max abs diff={diff.max():.5f}  rel-err p50={np.median(rel):.2e} p999={np.quantile(rel, 0.999):.2e}")

t_gpu = bench(lambda: mx.eval(orig_call(mlp, x)))
t_hook = bench(lambda: mx.eval(mlp(x)))
print(f"GPU-only: {t_gpu*1e3:.1f} ms   hooked: {t_hook*1e3:.1f} ms  ({(1 - t_hook / t_gpu) * 100:+.1f}%)")

Qwen3_5MLP.__call__ = orig_call
print("OK")
