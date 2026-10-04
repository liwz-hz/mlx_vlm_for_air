"""CPU/GPU co-execution for prefill MLP projections.

During prefill (compute-bound), splits MLP gate/up/down projections by
output dimension: GPU handles ~85%, CPU (Accelerate BLAS/AMX) handles ~15%
in parallel. Uses zero-copy memoryview on Apple Silicon unified memory.

Enable via MLX_VLM_CPU_PREFILL=<fraction> (e.g. 0.15); default disabled.
"""

import logging
import os
from concurrent.futures import ThreadPoolExecutor

import mlx.core as mx
import mlx.nn as nn
import numpy as np

logger = logging.getLogger(__name__)

_fraction_val = None
_executor = None
_weight_cache = {}
_stats = {"calls": 0, "prefill_splits": 0}
_orig_mlp_call = None
_installed = False


def _fraction():
    global _fraction_val
    if _fraction_val is None:
        _fraction_val = float(os.environ.get("MLX_VLM_CPU_PREFILL", "0"))
    return _fraction_val


def _enabled():
    return 0.0 < _fraction() <= 0.4 and not _installed


def _get_executor():
    global _executor
    if _executor is None:
        _executor = ThreadPoolExecutor(max_workers=1)
    return _executor


def _mx_view(arr):
    """Zero-copy numpy view of an evaluated MLX fp32 array (unified memory)."""
    arr = arr.astype(mx.float32)
    mx.eval(arr)
    mv = memoryview(arr)
    return np.frombuffer(mv, dtype=np.float32).reshape(arr.shape)


def _get_cpu_weights(linear, n_cpu):
    """Dequantized CPU weight rows as numpy fp32 (zero-copy view). Cached."""
    key = id(linear)
    cached = _weight_cache.get(key)
    if cached is not None and cached[0] is linear and cached[1] == n_cpu:
        return cached[2]

    w_deq = mx.dequantize(
        linear["weight"][:n_cpu],
        linear["scales"][:n_cpu],
        linear["biases"][:n_cpu],
        group_size=linear.group_size,
        bits=linear.bits,
    )
    w_np = _mx_view(w_deq)
    _weight_cache[key] = (linear, n_cpu, w_np)
    return w_np


def _split_projection(linear, x, fraction):
    """Split a quantized projection between GPU and CPU by output rows."""
    N = linear["weight"].shape[0]
    n_cpu = int(N * fraction) & ~7
    if n_cpu < 16 or N - n_cpu < 16:
        return linear(x)

    B, S, K = x.shape
    x_2d = x.reshape(B * S, K)

    x_np = _mx_view(x_2d)
    w_np = _get_cpu_weights(linear, n_cpu)

    def cpu_work():
        return x_np @ w_np.T

    fut = _get_executor().submit(cpu_work)

    N_gpu = N - n_cpu
    gpu_out = mx.quantized_matmul(
        x_2d,
        linear["weight"][n_cpu:],
        scales=linear["scales"][n_cpu:],
        biases=linear["biases"][n_cpu:],
        transpose=True,
        group_size=linear.group_size,
        bits=linear.bits,
        mode=linear.mode,
    )
    mx.eval(gpu_out)

    cpu_out = fut.result()
    cpu_mx = mx.array(cpu_out.tolist()).astype(x.dtype)
    cpu_mx = cpu_mx.reshape(*x.shape[:-1], n_cpu)
    gpu_mx = gpu_out.reshape(*x.shape[:-1], N_gpu)

    result = mx.concatenate([cpu_mx, gpu_mx], axis=-1)
    mx.eval(result)
    _stats["prefill_splits"] += 1
    return result


def _swiglu(gate, up):
    from ..models.activations import swiglu
    return swiglu(gate, up)


def _hybrid_mlp_call(self, x):
    """Replacement for Qwen3_5MLP.__call__ with CPU/GPU prefill split."""
    _stats["calls"] += 1
    frac = _fraction()
    S = x.shape[1] if x.ndim == 3 else x.shape[0]

    if frac <= 0 or S <= 8 or not isinstance(self.gate_proj, nn.QuantizedLinear):
        return _orig_mlp_call(self, x)

    try:
        gate = _split_projection(self.gate_proj, x, frac)
        up = _split_projection(self.up_proj, x, frac)
        hidden = _swiglu(gate, up)
        down = _split_projection(self.down_proj, hidden, frac)
        return down
    except Exception as e:
        logger.warning("cpu_prefill fallback: %s", e)
        return _orig_mlp_call(self, x)


def install():
    global _orig_mlp_call, _installed
    if _installed or _fraction() <= 0:
        return
    from ..models.qwen3_5.language import Qwen3_5MLP
    _orig_mlp_call = Qwen3_5MLP.__call__
    Qwen3_5MLP.__call__ = _hybrid_mlp_call
    _installed = True
    logger.info("cpu_prefill installed: fraction=%.2f", _fraction())


def maybe_install():
    if _enabled():
        install()
