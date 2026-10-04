"""CPU/GPU co-execution for prefill — universal QuantizedLinear M-split v2.

Only splits LARGE projections (M×N > 50M) to avoid overhead on small ones.
Weight cache sized for one full layer (~2GB). Covers MLP gate/up/down,
delta-net in_proj_qkv/z, attention q/o — the projections that matter.

Enable via MLX_VLM_CPU_PREFILL=<fraction> (e.g. 0.10); default disabled.
"""

import logging
import os
import ctypes
from collections import OrderedDict
from concurrent.futures import ThreadPoolExecutor

import mlx.core as mx
import mlx.nn as nn
import numpy as np

logger = logging.getLogger(__name__)

_fraction_val = None
_executor = None
_orig_ql_call = None
_installed = False
_weight_cache = OrderedDict()
_cache_bytes = 0
_CACHE_LIMIT = 4 << 30  # 4 GB
_MIN_COMPUTE = 50_000_000  # only split if M × N > 50M
_stats = {"calls": 0, "splits": 0, "skipped_small": 0}

_swiglu_lib = None
def _get_swiglu_lib():
    global _swiglu_lib
    if _swiglu_lib is None:
        try:
            import pathlib
            p = pathlib.Path(__file__).parent / "libswiglu_neon.dylib"
            if p.exists():
                _swiglu_lib = ctypes.CDLL(str(p))
                _swiglu_lib.swiglu_neon.argtypes = [
                    ctypes.POINTER(ctypes.c_float)] * 3 + [ctypes.c_int]
            else:
                _swiglu_lib = False
        except Exception:
            _swiglu_lib = False
    return _swiglu_lib or None


def _fraction():
    global _fraction_val
    if _fraction_val is None:
        _fraction_val = float(os.environ.get("MLX_VLM_CPU_PREFILL", "0"))
    return _fraction_val


def _enabled():
    return 0.0 < _fraction() <= 0.3 and not _installed


def _get_executor():
    global _executor
    if _executor is None:
        _executor = ThreadPoolExecutor(max_workers=1)
    return _executor


def _get_weights(linear):
    """Dequantized fp32 numpy view. LRU cache, 4GB limit."""
    global _cache_bytes
    key = id(linear)
    if key in _weight_cache:
        _weight_cache.move_to_end(key)
        return _weight_cache[key]

    w = mx.dequantize(
        linear["weight"], linear["scales"], linear["biases"],
        group_size=linear.group_size, bits=linear.bits,
    )
    w_f32 = w.astype(mx.float32)
    mx.eval(w_f32)
    K = linear["weight"].shape[1] * 32 // linear.bits
    w_np = np.frombuffer(memoryview(w_f32), dtype=np.float32).reshape(-1, K)

    nbytes = w_np.nbytes
    while _cache_bytes + nbytes > _CACHE_LIMIT and _weight_cache:
        _, old = _weight_cache.popitem(last=False)
        _cache_bytes -= old.nbytes
    _weight_cache[key] = w_np
    _cache_bytes += nbytes
    return w_np


def _hybrid_ql_call(self, x):
    """M-split for large QuantizedLinear projections during prefill."""
    _stats["calls"] += 1
    frac = _fraction()

    if x.ndim == 3:
        B, S, K = x.shape
        M = B * S
    elif x.ndim == 2:
        M, K = x.shape
        B, S = 1, M
    else:
        return _orig_ql_call(self, x)

    N = self["weight"].shape[0]

    if frac <= 0 or M <= 32:
        return _orig_ql_call(self, x)

    # 只拆大投影：小投影的开销大于收益
    if M * N < _MIN_COMPUTE:
        _stats["skipped_small"] += 1
        return _orig_ql_call(self, x)

    M_cpu = max(8, int(M * frac)) & ~7
    M_gpu = M - M_cpu
    if M_gpu < 16:
        return _orig_ql_call(self, x)

    try:
        if x.ndim == 3:
            x_2d = x.reshape(M, K)
        else:
            x_2d = x

        x_mv = memoryview(x_2d)
        w_np = _get_weights(self)

        x_gpu = x_2d[:M_gpu]
        gpu_out = _orig_ql_call(self, x_gpu)

        def cpu_work():
            x_u16 = np.frombuffer(x_mv, dtype=np.uint16).reshape(M, K)[M_gpu:]
            xn = (x_u16.astype(np.uint32) << 16).view(np.float32).reshape(M_cpu, K)
            return xn @ w_np.T

        fut = _get_executor().submit(cpu_work)
        mx.eval(gpu_out)

        cpu_res = fut.result()
        cpu_mx = mx.array(cpu_res).astype(x.dtype).reshape(M_cpu, N)
        gpu_2d = gpu_out.reshape(M_gpu, N)
        result = mx.concatenate([gpu_2d, cpu_mx], axis=0)

        if x.ndim == 3:
            result = result.reshape(B, S, N)
        _stats["splits"] += 1
        return result

    except Exception as e:
        logger.warning("cpu_prefill fallback: %s", e)
        return _orig_ql_call(self, x)


def install():
    global _orig_ql_call, _installed
    if _installed or _fraction() <= 0:
        return
    _orig_ql_call = nn.QuantizedLinear.__call__
    nn.QuantizedLinear.__call__ = _hybrid_ql_call
    _installed = True
    logger.info("cpu_prefill_v2 installed: frac=%.2f, min_compute=%dM",
                _fraction(), _MIN_COMPUTE // 1_000_000)


def maybe_install():
    if _enabled():
        install()
