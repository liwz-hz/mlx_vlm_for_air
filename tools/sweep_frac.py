"""frac sweep for the MLP K-split hook (find CPU/GPU balance point)."""
import os
import sys
import time

import mlx.core as mx
import mlx.nn as nn
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from mlx_vlm import cpu_prefill
from mlx_vlm.models.qwen3_5.language import Qwen3_5GatedDeltaNet, Qwen3_5MLP

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
Qwen3_5MLP.__call__ = cpu_prefill._hybrid_mlp_call
cpu_prefill._orig_mlp_call = orig_call

def bench(fn, reps=4):
    fn()
    ts = []
    for _ in range(reps):
        t0 = time.perf_counter()
        fn()
        ts.append(time.perf_counter() - t0)
    return min(ts)

t_gpu = bench(lambda: mx.eval(mlp(x)))
print(f"GPU-only: {t_gpu*1e3:.1f} ms")

for frac in (0.13, 0.12, 0.11, 0.10, 0.09, 0.08, 0.07, 0.09):
    cpu_prefill._fraction_val = frac
    out = mlp(x)
    mx.eval(out)
    t = bench(lambda: mx.eval(mlp(x)))
    t_gpu2 = bench(lambda: mx.eval(orig_call(mlp, x)))
    print(f"frac={frac:.3f}: {t*1e3:7.1f} ms  (gpu-now {t_gpu2*1e3:.1f})  ({(1 - t / t_gpu2) * 100:+.1f}%)")

Qwen3_5MLP.__call__ = orig_call
