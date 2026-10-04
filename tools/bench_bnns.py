"""Benchmark BNNS fully-connected GEMM (AMX) in fp16/bf16/fp32 via ctypes.

out[m, :] = W @ in[m, :] for batch M rows == GEMM out = x @ W.T
"""
import ctypes
import time
import numpy as np

LIB = "/System/Library/Frameworks/Accelerate.framework/Versions/A/Frameworks/vecLib.framework/libBNNS.dylib"
lib = ctypes.CDLL(LIB)

BNNS_MAX_TENSOR_DIMENSION = 8


class BNNSNDArrayDescriptor(ctypes.Structure):
    _fields_ = [
        ("flags", ctypes.c_uint32),
        ("layout", ctypes.c_uint32),
        ("size", ctypes.c_size_t * BNNS_MAX_TENSOR_DIMENSION),
        ("stride", ctypes.c_size_t * BNNS_MAX_TENSOR_DIMENSION),
        ("data", ctypes.c_void_p),
        ("data_type", ctypes.c_uint32),
        ("table_data", ctypes.c_void_p),
        ("table_data_type", ctypes.c_uint32),
        ("data_scale", ctypes.c_float),
        ("data_bias", ctypes.c_float),
    ]


class BNNSActivation(ctypes.Structure):
    _fields_ = [
        ("function", ctypes.c_int32),
        ("alpha", ctypes.c_float),
        ("beta", ctypes.c_float),
        ("iscale", ctypes.c_int32),
        ("ioffset", ctypes.c_int32),
        ("ishift", ctypes.c_int32),
        ("iscale_per_channel", ctypes.c_void_p),
        ("ioffset_per_channel", ctypes.c_void_p),
        ("ishift_per_channel", ctypes.c_void_p),
    ]


class BNNSLayerParametersFullyConnected(ctypes.Structure):
    _fields_ = [
        ("i_desc", BNNSNDArrayDescriptor),
        ("w_desc", BNNSNDArrayDescriptor),
        ("o_desc", BNNSNDArrayDescriptor),
        ("bias", BNNSNDArrayDescriptor),
        ("activation", BNNSActivation),
    ]


class BNNSFilterParameters(ctypes.Structure):
    _fields_ = [
        ("flags", ctypes.c_uint32),
        ("n_threads", ctypes.c_size_t),
        ("alloc_memory", ctypes.c_void_p),
        ("free_memory", ctypes.c_void_p),
    ]


DT = {
    "fp16": 0x10000 | 16,
    "fp32": 0x10000 | 32,
    "bf16": 0x10000 | 0x8000 | 16,
}
LAYOUT_VECTOR = 0x10000
LAYOUT_ROWMAJOR = 0x20000

lib.BNNSFilterCreateLayerFullyConnected.restype = ctypes.c_void_p
lib.BNNSFilterCreateLayerFullyConnected.argtypes = [
    ctypes.POINTER(BNNSLayerParametersFullyConnected),
    ctypes.POINTER(BNNSFilterParameters),
]
lib.BNNSFilterApplyBatch.restype = ctypes.c_int
lib.BNNSFilterApplyBatch.argtypes = [
    ctypes.c_void_p,
    ctypes.c_size_t,
    ctypes.c_void_p,
    ctypes.c_size_t,
    ctypes.c_void_p,
    ctypes.c_size_t,
]
lib.BNNSFilterDestroy.argtypes = [ctypes.c_void_p]


def desc_vec(n, dtype, data=None):
    d = BNNSNDArrayDescriptor()
    d.flags = 0
    d.layout = LAYOUT_VECTOR
    d.size[0] = n
    d.stride[0] = 0
    d.data = data
    d.data_type = DT[dtype]
    return d


def desc_mat(rows, cols, dtype, data):
    d = BNNSNDArrayDescriptor()
    d.flags = 0
    d.layout = LAYOUT_ROWMAJOR
    d.size[0] = rows
    d.size[1] = cols
    d.stride[0] = 0
    d.stride[1] = 0
    d.data = data
    d.data_type = DT[dtype]
    return d


def make_filter(k, n, w_dtype, in_dtype, out_dtype, w_ptr, n_threads=0):
    params = BNNSLayerParametersFullyConnected()
    params.i_desc = desc_vec(k, in_dtype)
    params.w_desc = desc_mat(n, k, w_dtype, w_ptr)
    params.o_desc = desc_vec(n, out_dtype)
    params.bias = BNNSNDArrayDescriptor()  # data NULL -> no bias
    params.activation = BNNSActivation()
    params.activation.function = 0  # identity
    fp = BNNSFilterParameters()
    fp.flags = 0
    fp.n_threads = n_threads
    f = lib.BNNSFilterCreateLayerFullyConnected(
        ctypes.byref(params), ctypes.byref(fp)
    )
    return f


def bench_bnns(M, K, N, w_dtype, in_dtype, out_dtype, check=True, reps=3):
    rng = np.random.default_rng(0)
    w32 = (rng.standard_normal((N, K)) * 0.02).astype(np.float32)
    x32 = (rng.standard_normal((M, K)) * 0.5).astype(np.float32)

    def to(a, dt):
        return {"fp16": a.astype(np.float16), "fp32": a, "bf16": a.astype(")</parameter>").astype(np.uint16) if False else a.astype(np.float16)}[dt]

    # bf16: bit-trick round from fp32
    def f32_to_bf16(a):
        u = a.view(np.uint32)
        r = (u + np.uint32(0x7FFF) + ((u >> np.uint32(16)) & np.uint32(1))) >> np.uint32(16)
        return r.astype(np.uint16)

    def as_dt(a, dt):
        if dt == "fp32":
            return a
        if dt == "fp16":
            return a.astype(np.float16)
        return f32_to_bf16(a)

    w = np.ascontiguousarray(as_dt(w32, w_dtype))
    x = np.ascontiguousarray(as_dt(x32, in_dtype))
    out = np.zeros((M, N), dtype=np.float32 if out_dtype == "fp32" else (np.float16 if out_dtype == "fp16" else np.uint16))

    f = make_filter(K, N, w_dtype, in_dtype, out_dtype, w.ctypes.data, n_threads=0)
    if not f:
        print(f"  {w_dtype}/{in_dtype}/{out_dtype}: create FAILED")
        return None
    rc = lib.BNNSFilterApplyBatch(
        f, M, x.ctypes.data, K, out.ctypes.data, N
    )
    if rc != 0:
        print(f"  {w_dtype}/{in_dtype}/{out_dtype}: apply rc={rc}")
        lib.BNNSFilterDestroy(f)
        return None
    ts = []
    for _ in range(reps):
        t0 = time.perf_counter()
        rc = lib.BNNSFilterApplyBatch(f, M, x.ctypes.data, K, out.ctypes.data, N)
        ts.append(time.perf_counter() - t0)
    t = min(ts)
    tf = 2 * M * K * N / t / 1e12
    err = ""
    if check:
        ref = x32 @ w32.T
        got = out.astype(np.float32)
        err = f" maxabs={np.abs(got - ref).max():.4f}"
    print(f"  {w_dtype}/{in_dtype}/{out_dtype}: {t*1e3:7.1f} ms  {tf:5.2f} TFLOPS{err}")
    lib.BNNSFilterDestroy(f)
    return t


M, K, N = 2048, 5120, 2090
print(f"GEMM M={M} K={K} N={N} (MLP gate CPU-share shape):")
for wdt, idt, odt in [
    ("fp32", "fp32", "fp32"),
    ("fp16", "fp16", "fp32"),
    ("fp16", "fp16", "fp16"),
    ("bf16", "bf16", "fp32"),
    ("bf16", "bf16", "bf16"),
]:
    try:
        bench_bnns(M, K, N, wdt, idt, odt)
    except Exception as e:
        print(f"  {wdt}/{idt}/{odt}: EXC {e}")

M2, K2, N2 = 2048, 2176, 5120
print(f"GEMM M={M2} K={K2} N={N2} (MLP down K-split shape):")
for wdt, idt, odt in [("fp32", "fp32", "fp32"), ("bf16", "bf16", "fp32"), ("fp16", "fp16", "fp32")]:
    try:
        bench_bnns(M2, K2, N2, wdt, idt, odt)
    except Exception as e:
        print(f"  {wdt}/{idt}/{odt}: EXC {e}")
