"""CPU/GPU co-execution for the speculative verify path.

Splits each verify projection's output rows between GPU (majority, the
fork's exact kernel) and CPU (minority, a NEON-tiled 4-bit GEMV in
mlx_vlm/cpu_qmv.c). The GPU kernel is ALU-bound at T=4 while ~35% of its
memory bandwidth sits idle; the CPU worker fills that idle bandwidth.

Enabled via MLX_VLM_CPU_SPLIT=<fraction of rows for CPU, e.g. 0.16>;
default disabled (0).
"""

import ctypes
import logging
import os
import threading
from concurrent.futures import ThreadPoolExecutor

import mlx.core as mx
import numpy as np

logger = logging.getLogger(__name__)

_LIB = None
_THREADS = 6
_executor = None
_cpu_cache = {}


def _load_lib():
    global _LIB
    if _LIB is not None:
        return _LIB
    try:
        import pathlib

        lib_path = pathlib.Path(__file__).parent / "libcpu_qmv.dylib"
        if not lib_path.exists():
            return None
        lib = ctypes.CDLL(str(lib_path))
        lib.cpu_qmv.argtypes = [
            ctypes.c_int, ctypes.c_int, ctypes.c_int,
            ctypes.POINTER(ctypes.c_float), ctypes.POINTER(ctypes.c_uint32),
            ctypes.POINTER(ctypes.c_float), ctypes.POINTER(ctypes.c_float),
            ctypes.POINTER(ctypes.c_float), ctypes.c_int,
        ]
        _LIB = lib
    except Exception as e:
        logger.warning("cpu_split unavailable: %s", e)
        _LIB = False
    return _LIB or None


def _fraction():
    try:
        v = float(os.environ.get("MLX_VLM_CPU_SPLIT", "0"))
    except ValueError:
        return 0.0
    return v


def _enabled():
    return 0.0 < _fraction() <= 0.4 and _load_lib() is not None


def _bf16_to_f32_numpy(arr):
    u16 = np.from_dlpack(arr.astype(mx.uint16).reshape(-1)) if False else None
    raw = np.from_dlpack(arr)
    u16 = raw.view(np.uint16) if raw.dtype == np.uint16 else np.asarray(raw).view(np.uint16)
    u32 = u16.astype(np.uint32) << 16
    return u32.view(np.float32).reshape(arr.shape)


def _f32_from_bf16(arr):
    return arr.astype(mx.float32)


class _CpuSlice:
    __slots__ = ("wq", "scales", "biases", "n_cpu")


def _prepare(linear, n_cpu):
    key = id(linear)
    cached = _cpu_cache.get(key)
    if cached is not None and cached[0] is linear and cached[1].n_cpu == n_cpu:
        return cached[1]
    sl = _CpuSlice()
    wq_full = linear["weight"]
    mx.eval(wq_full)
    sl.wq = np.ascontiguousarray(np.from_dlpack(wq_full)[:n_cpu])
    s32 = linear["scales"][:n_cpu].astype(mx.float32)
    b32 = linear["biases"][:n_cpu].astype(mx.float32)
    mx.eval(s32, b32)
    sl.scales = np.ascontiguousarray(np.from_dlpack(s32))
    sl.biases = np.ascontiguousarray(np.from_dlpack(b32))
    sl.n_cpu = n_cpu
    _cpu_cache[key] = (linear, sl)
    return sl


def _cpu_run(sl, x32_np, T, K):
    lib = _load_lib()
    out = np.zeros((T, sl.n_cpu), dtype=np.float32)
    lib.cpu_qmv(
        T, K, sl.n_cpu,
        x32_np.ctypes.data_as(ctypes.POINTER(ctypes.c_float)),
        sl.wq.ctypes.data_as(ctypes.POINTER(ctypes.c_uint32)),
        sl.scales.ctypes.data_as(ctypes.POINTER(ctypes.c_float)),
        sl.biases.ctypes.data_as(ctypes.POINTER(ctypes.c_float)),
        out.ctypes.data_as(ctypes.POINTER(ctypes.c_float)),
        _THREADS,
    )
    return out


def gpu_rows_kernel(linear, x, n_cpu):
    from .models.quantized_verifier import _target_verify_qmv_kernel

    B, T, K = x.shape
    N = linear.weight.shape[0] - n_cpu
    kernel = _target_verify_qmv_kernel(4, 64, x.dtype, T, K, N)
    return kernel(
        inputs=[
            x,
            linear["weight"][n_cpu:],
            linear["scales"][n_cpu:],
            linear["biases"][n_cpu:],
        ],
        template=[
            ("T", x.dtype),
            ("VERIFY_T", int(T)),
            ("K_SIZE", int(K)),
            ("N_SIZE", int(N)),
        ],
        grid=(32, 2 * (N // 8), B),
        threadgroup=(32, 2, 1),
        output_shapes=[(B, T, N)],
        output_dtypes=[x.dtype],
    )[0]


def hybrid_eligible(linear, x):
    return (
        _enabled()
        and isinstance(linear.weight, mx.array)
        and linear.mode == "affine"
        and linear.bits == 4
        and linear.group_size == 64
        and "bias" not in linear
        and x.dtype == mx.bfloat16
        and x.ndim == 3
        and x.shape[0] == 1
        and 2 <= x.shape[1] <= 5
        and x.shape[-1] % 512 == 0
        and linear.weight.shape[0] % 16 == 0
    )


def hybrid_split_run(linear, x):
    B, T, K = x.shape
    N = linear.weight.shape[0]
    frac = _fraction()
    n_cpu = int(N * frac) & ~7
    if n_cpu < 16 or N - n_cpu < 16:
        return None
    sl = _prepare(linear, n_cpu)
    x32 = x.reshape(T, K).astype(mx.float32)
    mx.eval(x32)
    x32_np = np.from_dlpack(x32)

    global _executor
    if _executor is None:
        _executor = ThreadPoolExecutor(max_workers=1)
    fut = _executor.submit(_cpu_run, sl, x32_np, T, K)

    gpu_out = gpu_rows_kernel(linear, x, n_cpu)
    cpu_out = fut.result()

    cpu_mx = mx.array(cpu_out).astype(mx.bfloat16).reshape(1, T, n_cpu)
    out = mx.concatenate([cpu_mx, gpu_out], axis=-1)
    return out
