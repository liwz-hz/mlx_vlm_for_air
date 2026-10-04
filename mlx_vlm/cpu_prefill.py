"""CPU/GPU co-execution v3.1 for prefill.

Bound-point analysis (M5 Air, Qwen3.5-27B-4bit, prefill M>=512):
quantized GEMMs run at 11.2-12.2 TFLOPS (92-95% of GPU peak) and cover
~92% of prefill time, so the only lever on them is adding the CPU's
~1.7 TFLOPS (AMX). Per-op glue (casts, concats, syncs) eats most of the
benefit on small projections, so v3.1 splits only:

  * MLP — K-split, zero concats: gate/up are row-split (GPU computes
    columns [0, I-nc), CPU the rest via BNNSMatMul bf16 straight from
    the bf16 MLX buffer — no fp32 cast); the CPU applies NEON swiglu to
    its own columns and runs the matching K-slice of down_proj via
    sgemm, writing fp32 directly into an MLX buffer. The GPU combines
    with a single add: down_gpu + cpu.astype(bf16).
Every other projection (delta-net qkv/z/out, attention q/o) was measured
NEGATIVE: their consumer ops depend on the split output, so the GPU idles
while the CPU finishes — the offload benefit (<2.8ms) never covers the
concat+sync glue. The MLP is the only op whose CPU share fully overlaps
its own GPU share.

Enable via MLX_VLM_CPU_PREFILL=<fraction> (e.g. 0.125); default off.
Decode (M < 512) is untouched.
"""

import ctypes
import logging
import os
import threading
from concurrent.futures import ThreadPoolExecutor

import mlx.core as mx
import mlx.nn as nn
import numpy as np

logger = logging.getLogger(__name__)

_MIN_M = 512
_pool = ThreadPoolExecutor(max_workers=1)
_fraction_val = None
_installed = False
_orig_mlp_call = None


def _fraction():
    global _fraction_val
    if _fraction_val is None:
        _fraction_val = float(os.environ.get("MLX_VLM_CPU_PREFILL", "0"))
    return _fraction_val


def _enabled():
    return 0.0 < _fraction() <= 0.35 and not _installed


def _lin_ok(lin):
    return (
        isinstance(lin, nn.QuantizedLinear)
        and lin.bits == 4
        and lin.group_size == 64
    )


def _np_view(arr, dtype, *shape):
    return np.frombuffer(memoryview(arr), dtype=dtype).reshape(*shape)


# ---------------------------------------------------------------------------
# BNNS bf16 GEMM (AMX): C[M,N] fp32 = A[M,K] bf16 @ B[N,K] bf16 ^T
# ---------------------------------------------------------------------------

_DT_BF16 = 0x10000 | 0x8000 | 16
_DT_F32 = 0x10000 | 32
_LAYOUT_2D_LAST_MAJOR = 0x28000

_MAX_DIM = 8

_bnns_lock = threading.Lock()
_bnns = None
_bnns_ws = {}


class _Desc(ctypes.Structure):
    _fields_ = [
        ("flags", ctypes.c_uint32),
        ("layout", ctypes.c_uint32),
        ("size", ctypes.c_size_t * _MAX_DIM),
        ("stride", ctypes.c_size_t * _MAX_DIM),
        ("data", ctypes.c_void_p),
        ("data_type", ctypes.c_uint32),
        ("table_data", ctypes.c_void_p),
        ("table_data_type", ctypes.c_uint32),
        ("data_scale", ctypes.c_float),
        ("data_bias", ctypes.c_float),
    ]


class _FilterParams(ctypes.Structure):
    _fields_ = [
        ("flags", ctypes.c_uint32),
        ("n_threads", ctypes.c_size_t),
        ("alloc_memory", ctypes.c_void_p),
        ("free_memory", ctypes.c_void_p),
    ]


def _bnns_init():
    global _bnns
    if _bnns is not None:
        return _bnns
    try:
        import ctypes.util

        lib = ctypes.CDLL(
            "/System/Library/Frameworks/Accelerate.framework/Versions/A/"
            "Frameworks/vecLib.framework/libBNNS.dylib"
        )
        lib.BNNSMatMulWorkspaceSize.restype = ctypes.c_ssize_t
        lib.BNNSMatMulWorkspaceSize.argtypes = [
            ctypes.c_bool,
            ctypes.c_bool,
            ctypes.c_float,
            ctypes.POINTER(_Desc),
            ctypes.POINTER(_Desc),
            ctypes.POINTER(_Desc),
            ctypes.POINTER(_FilterParams),
        ]
        lib.BNNSMatMul.restype = ctypes.c_int
        lib.BNNSMatMul.argtypes = [
            ctypes.c_bool,
            ctypes.c_bool,
            ctypes.c_float,
            ctypes.POINTER(_Desc),
            ctypes.POINTER(_Desc),
            ctypes.POINTER(_Desc),
            ctypes.c_void_p,
            ctypes.POINTER(_FilterParams),
        ]
        _bnns = lib
    except Exception:
        _bnns = False
        logger.warning("cpu_prefill: BNNS unavailable, co-exec disabled")
    return _bnns


def _desc2d(rows, cols, dtype, ptr):
    d = _Desc()
    d.flags = 0
    d.layout = _LAYOUT_2D_LAST_MAJOR
    d.size[0] = rows
    d.size[1] = cols
    d.stride[0] = cols  # row-major: last dim contiguous
    d.stride[1] = 1
    d.data = ptr
    d.data_type = dtype
    return d


def _bnns_gemm_bf16(a_u16, b_u16, c_f32):
    """C[M,N] fp32 = A[M,K] bf16 @ B[N,K] bf16 ^T. numpy views, any thread."""
    lib = _bnns or _bnns_init()
    if not lib:
        raise RuntimeError("BNNS unavailable")
    m, k = a_u16.shape
    n, k2 = b_u16.shape
    assert k == k2
    ad = _desc2d(m, k, _DT_BF16, a_u16.ctypes.data)
    bd = _desc2d(n, k, _DT_BF16, b_u16.ctypes.data)
    cd = _desc2d(m, n, _DT_F32, c_f32.ctypes.data)
    fp = _FilterParams()
    fp.n_threads = 0
    with _bnns_lock:
        ws = _bnns_ws.get((m, k, n))
        if ws is None:
            sz = lib.BNNSMatMulWorkspaceSize(
                False, True, 1.0, ctypes.byref(ad), ctypes.byref(bd),
                ctypes.byref(cd), ctypes.byref(fp),
            )
            if sz < 0:
                raise RuntimeError(f"BNNS workspace size {sz}")
            ws = np.empty(max(int(sz), 1), dtype=np.uint8)
            _bnns_ws[(m, k, n)] = ws
        rc = lib.BNNSMatMul(
            False, True, 1.0, ctypes.byref(ad), ctypes.byref(bd),
            ctypes.byref(cd), ws.ctypes.data, ctypes.byref(fp),
        )
    if rc != 0:
        raise RuntimeError(f"BNNSMatMul rc={rc}")


# ---------------------------------------------------------------------------
# NEON swiglu (libswiglu_neon.dylib; falls back to numpy)
# ---------------------------------------------------------------------------

_swiglu_lib = None


def _get_swiglu_lib():
    global _swiglu_lib
    if _swiglu_lib is None:
        try:
            import pathlib

            p = pathlib.Path(__file__).parent / "libswiglu_neon.dylib"
            if p.exists():
                lib = ctypes.CDLL(str(p))
                lib.swiglu_neon.argtypes = [
                    ctypes.POINTER(ctypes.c_float),
                    ctypes.POINTER(ctypes.c_float),
                    ctypes.POINTER(ctypes.c_float),
                    ctypes.c_int,
                ]
                _swiglu_lib = lib
            else:
                _swiglu_lib = False
        except Exception:
            _swiglu_lib = False
    return _swiglu_lib or None


def _swiglu_f32(gate, up):
    """gate*sigmoid(gate)*up, fp32, in/out contiguous numpy."""
    lib = _get_swiglu_lib()
    if lib is None:
        return gate * (1.0 / (1.0 + np.exp(-gate))) * up
    out = np.empty_like(gate)
    lib.swiglu_neon(
        gate.ctypes.data_as(ctypes.POINTER(ctypes.c_float)),
        up.ctypes.data_as(ctypes.POINTER(ctypes.c_float)),
        out.ctypes.data_as(ctypes.POINTER(ctypes.c_float)),
        gate.size,
    )
    return out


# ---------------------------------------------------------------------------
# CPU worker (single thread, zero MLX API calls)
# ---------------------------------------------------------------------------


_share_lib = None


def _get_share_lib():
    """Fused C library: one ctypes call per op -> GIL released for the whole
    CPU share (a Python-level loop measured +40% wall time via GIL contention
    with MLX's async dispatch thread)."""
    global _share_lib
    if _share_lib is None:
        try:
            import pathlib

            p = pathlib.Path(__file__).parent / "libcpu_share.dylib"
            if p.exists():
                lib = ctypes.CDLL(str(p))
                lib.mlp_cpu_share.restype = ctypes.c_int
                lib.mlp_cpu_share.argtypes = [
                    ctypes.POINTER(ctypes.c_uint16),
                    ctypes.POINTER(ctypes.c_uint16),
                    ctypes.POINTER(ctypes.c_uint16),
                    ctypes.POINTER(ctypes.c_float),
                    ctypes.POINTER(ctypes.c_float),
                    ctypes.c_size_t,
                    ctypes.c_size_t,
                    ctypes.c_size_t,
                    ctypes.c_size_t,
                ]
                lib.split_gemm_bf16.restype = None
                lib.split_gemm_bf16.argtypes = [
                    ctypes.POINTER(ctypes.c_uint16),
                    ctypes.POINTER(ctypes.c_uint16),
                    ctypes.POINTER(ctypes.c_float),
                    ctypes.c_size_t,
                    ctypes.c_size_t,
                    ctypes.c_size_t,
                ]
                _share_lib = lib
            else:
                _share_lib = False
        except Exception:
            _share_lib = False
    return _share_lib or None


def _cpu_mlp(x_u16, wg_u16, wu_u16, wd_f32, out_f32, m, nc):
    """MLP CPU share via one fused C call: BNNS gate/up (bf16, AMX) +
    NEON swiglu + down K-slice sgemm. All buffers are numpy views."""
    lib = _get_share_lib()
    u16p = ctypes.POINTER(ctypes.c_uint16)
    f32p = ctypes.POINTER(ctypes.c_float)
    if lib is not None:
        rc = lib.mlp_cpu_share(
            x_u16.ctypes.data_as(u16p),
            wg_u16.ctypes.data_as(u16p),
            wu_u16.ctypes.data_as(u16p),
            wd_f32.ctypes.data_as(f32p),
            out_f32.ctypes.data_as(f32p),
            m,
            x_u16.shape[1],
            nc,
            out_f32.shape[1],
        )
        if rc != 0:
            raise RuntimeError(f"mlp_cpu_share rc={rc}")
        return
    gate = np.empty((m, nc), dtype=np.float32)
    up = np.empty((m, nc), dtype=np.float32)
    _bnns_gemm_bf16(x_u16, wg_u16, gate)
    _bnns_gemm_bf16(x_u16, wu_u16, up)
    hidden = _swiglu_f32(gate, up)
    np.matmul(hidden, wd_f32.T, out=out_f32)


def _cpu_split_gemms(x_u16, jobs):
    """jobs: (w_u16 [nc,K] bf16, out_f32 [M,nc] view of MLX fp32 buffer)."""
    lib = _get_share_lib()
    u16p = ctypes.POINTER(ctypes.c_uint16)
    f32p = ctypes.POINTER(ctypes.c_float)
    for w_u16, out_f32 in jobs:
        if lib is not None:
            lib.split_gemm_bf16(
                x_u16.ctypes.data_as(u16p),
                w_u16.ctypes.data_as(u16p),
                out_f32.ctypes.data_as(f32p),
                x_u16.shape[0],
                x_u16.shape[1],
                w_u16.shape[0],
            )
        else:
            _bnns_gemm_bf16(x_u16, w_u16, out_f32)


# ---------------------------------------------------------------------------
# Hooks
# ---------------------------------------------------------------------------


def _hybrid_mlp_call(self, x):
    frac = _fraction()
    b, s, k = x.shape
    m = b * s
    if (
        frac <= 0
        or m < _MIN_M
        or not _bnns_init()
        or not _lin_ok(self.gate_proj)
        or not _lin_ok(self.up_proj)
        or not _lin_ok(self.down_proj)
    ):
        return _orig_mlp_call(self, x)

    try:
        from .models.activations import swiglu

        i = self.gate_proj["weight"].shape[0]  # intermediate size
        n_down = self.down_proj["weight"].shape[0]
        # nc: CPU share of the intermediate dim (multiple of 64 for group-
        # aligned K-slicing of down_proj's packed weights + scales)
        nc = max(64, int(round(i * frac))) & ~63
        if i - nc < 64 or m * nc < (1 << 18):
            return _orig_mlp_call(self, x)

        x2 = x.reshape(m, k)

        # --- light GPU phase: dequant CPU weights + output buffer ---
        wg_bf = mx.dequantize(
            self.gate_proj["weight"][i - nc:],
            self.gate_proj["scales"][i - nc:],
            self.gate_proj["biases"][i - nc:],
            group_size=64, bits=4,
        ).astype(mx.bfloat16)
        wu_bf = mx.dequantize(
            self.up_proj["weight"][i - nc:],
            self.up_proj["scales"][i - nc:],
            self.up_proj["biases"][i - nc:],
            group_size=64, bits=4,
        ).astype(mx.bfloat16)
        # down_proj K-slice (CPU gets the LAST nc columns of W):
        # weight [N_down, I/8] packed, scales [N_down, I/64]
        wd_f32 = mx.dequantize(
            self.down_proj["weight"][:, (i - nc) // 8 :],
            self.down_proj["scales"][:, (i - nc) // 64 :],
            self.down_proj["biases"][:, (i - nc) // 64 :],
            group_size=64, bits=4,
        ).astype(mx.float32)
        out_buf = mx.empty((m, n_down), dtype=mx.float32)
        mx.async_eval([wg_bf, wu_bf, wd_f32, out_buf])

        x_u16 = _np_view(x2, np.uint16, m, k)
        wg_u16 = _np_view(wg_bf, np.uint16, nc, k)
        wu_u16 = _np_view(wu_bf, np.uint16, nc, k)
        wd_np = _np_view(wd_f32, np.float32, n_down, nc)
        out_np = _np_view(out_buf, np.float32, m, n_down)
        # x2 must be materialized for the CPU to read
        mx.eval(x2)

        # --- heavy GPU phase: gate/up rows, swiglu, down K-slice ---
        gate_gpu = mx.quantized_matmul(
            x2, self.gate_proj["weight"][: i - nc],
            self.gate_proj["scales"][: i - nc],
            self.gate_proj["biases"][: i - nc],
            transpose=True, group_size=64, bits=4,
        )
        up_gpu = mx.quantized_matmul(
            x2, self.up_proj["weight"][: i - nc],
            self.up_proj["scales"][: i - nc],
            self.up_proj["biases"][: i - nc],
            transpose=True, group_size=64, bits=4,
        )
        hidden_gpu = swiglu(gate_gpu, up_gpu)
        down_gpu = mx.quantized_matmul(
            hidden_gpu,
            self.down_proj["weight"][:, : (i - nc) // 8],
            self.down_proj["scales"][:, : (i - nc) // 64],
            self.down_proj["biases"][:, : (i - nc) // 64],
            transpose=True, group_size=64, bits=4,
        )
        mx.async_eval([down_gpu])

        fut = _pool.submit(_cpu_mlp, x_u16, wg_u16, wu_u16, wd_np, out_np, m, nc)
        fut.result()
        return (down_gpu + out_buf.astype(mx.bfloat16)).reshape(b, s, k)
    except Exception:
        logger.warning("cpu_prefill mlp fallback", exc_info=True)
        return _orig_mlp_call(self, x)


def install():
    global _orig_mlp_call, _installed
    if _installed or _fraction() <= 0:
        return
    if not _bnns_init():
        return
    from .models.qwen3_5.language import Qwen3_5MLP

    _orig_mlp_call = Qwen3_5MLP.__call__
    Qwen3_5MLP.__call__ = _hybrid_mlp_call
    _installed = True
    logger.info(
        "cpu_prefill v3.1 installed: frac=%.3f (MLP K-split only)",
        _fraction(),
    )


def maybe_install():
    if _enabled():
        install()
