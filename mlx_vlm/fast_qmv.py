"""Fast skinny-M (2..8) affine 4-bit quantized matmul for Apple Silicon.

Routes nn.QuantizedLinear calls with small row counts (MTP speculative
verification batches) through a custom Metal kernel using algebraic
scale/bias folding and half2 FMA. Enabled for the server via
MLX_VLM_FAST_QMV (default "0", experimental); set "1" to enable.

Note: the speculative verify hot path uses mlx_vlm/models/quantized_verifier.py
custom kernels and bypasses nn.QuantizedLinear, so this patch mainly affects
the drafter per-round layer pass (~5% of round time). The drafter input
projection (K=10240, concat(hidden, embedding)) MUST be excluded (done):
routing it through the fp16 fast path corrupts drafts (embedding magnitudes
overflow fp16) and collapses speculative acceptance.
"""

import logging
import os

import mlx.core as mx
import mlx.nn as nn

logger = logging.getLogger(__name__)

_SOURCE = """
    const int row = threadgroup_position_in_grid.x;
    const int lane = thread_position_in_threadgroup.x % 32;
    const int sg = simdgroup_index_in_threadgroup;
    const int n_groups = K / GROUP;
    const int stride = 32 * NSG;

    float acc[M];
    for (int m = 0; m < M; m++) acc[m] = 0.0f;

    const device uint* wrow = w + (size_t)row * (K / 8);
    const device half* srow = s + (size_t)row * n_groups;
    const device half* brow = b + (size_t)row * n_groups;

    for (int g = sg * 32 + lane; g < n_groups; g += stride) {
        const float sc = float(srow[g]);
        const float bi = float(brow[g]);
        const device uint* wp = wrow + g * (GROUP / 8);
        half2 accn[M];
        #pragma unroll
        for (int m = 0; m < M; m++) accn[m] = half2(0.0h, 0.0h);
        for (int c = 0; c < GROUP / 8; c++) {
            uint packed = wp[c];
            int base = g * GROUP + c * 8;
            half2 nh[4];
            #pragma unroll
            for (int i = 0; i < 4; i++) {
                uint byte = (packed >> (8 * i)) & 0xFF;
                nh[i] = half2(half(byte & 0xF), half((byte >> 4) & 0xF));
            }
            #pragma unroll
            for (int m = 0; m < M; m++) {
                uint4 xw = *((const device uint4*)(x + (size_t)m * K + base));
                half2 x0 = as_type<half2>(xw.x);
                half2 x1 = as_type<half2>(xw.y);
                half2 x2 = as_type<half2>(xw.z);
                half2 x3 = as_type<half2>(xw.w);
                accn[m] = fma(nh[0], x0, accn[m]);
                accn[m] = fma(nh[1], x1, accn[m]);
                accn[m] = fma(nh[2], x2, accn[m]);
                accn[m] = fma(nh[3], x3, accn[m]);
            }
        }
        #pragma unroll
        for (int m = 0; m < M; m++) {
            float dots = float(accn[m].x) + float(accn[m].y);
            acc[m] = fma(sc, dots, acc[m]);
            acc[m] = fma(bi, xsum[(size_t)m * n_groups + g], acc[m]);
        }
    }

    threadgroup float partials[NSG][M];
    for (int m = 0; m < M; m++) {
        float v = simd_sum(acc[m]);
        if (lane == 0) partials[sg][m] = v;
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    if (thread_position_in_threadgroup.x == 0) {
        for (int m = 0; m < M; m++) {
            float total = 0.0f;
            #pragma unroll
            for (int i = 0; i < NSG; i++) total += partials[i][m];
            out[(size_t)m * N + row] = half(total);
        }
    }
"""

_NSG = 4
_stats = {"fast": 0, "fallback": 0}
_kernel = None
_conv_cache = {}
_orig_call = None
_hit_logged = set()
_TRACE = os.environ.get("MLX_VLM_FAST_QMV_TRACE") == "1"


def _get_kernel():
    global _kernel
    if _kernel is None:
        _kernel = mx.fast.metal_kernel(
            name="fast_qmv",
            input_names=["x", "w", "s", "b", "xsum", "K", "N"],
            output_names=["out"],
            source=_SOURCE,
        )
    return _kernel


def _fast_call(self, x):
    global _hit_logged
    if _TRACE and len(_hit_logged) < 60:
        try:
            _kd = self["weight"].shape[1] * 8
            _t = ("trace", x.size // _kd, _kd, self["weight"].shape[0], str(x.dtype))
            if _t not in _hit_logged:
                _hit_logged.add(_t)
                logger.info(
                    "fast_qmv trace: M=%d K=%d N=%d %s",
                    _t[1], _t[2], _t[3], _t[4],
                )
        except Exception:
            pass
    try:
        if (
            self.mode == "affine"
            and self.bits == 4
            and self.group_size == 64
            and "bias" not in self
            and x.dtype in (mx.float16, mx.bfloat16)
            and x.ndim >= 2
        ):
            K = self["weight"].shape[1] * 8
            if K % 64 == 0 and K != 10240:
                M = x.size // K
                if 2 <= M <= 8 and M * K == x.size:
                    N = self["weight"].shape[0]
                    n_groups = K // 64
                    x2 = x.reshape(M, K)
                    if x2.dtype == mx.bfloat16:
                        x16 = x2.astype(mx.float16)
                    else:
                        x16 = x2
                    key = id(self)
                    cached = _conv_cache.get(key)
                    if cached is None or cached[0] is not self:
                        s16 = self["scales"].astype(mx.float16)
                        b16 = self["biases"].astype(mx.float16)
                        _conv_cache[key] = (self, s16, b16)
                        cached = _conv_cache[key]
                    _, s16, b16 = cached
                    xsum = mx.sum(
                        x16.reshape(M, n_groups, 64).astype(mx.float32), axis=-1
                    )
                    out = _get_kernel()(
                        inputs=[x16, self["weight"], s16, b16, xsum, K, N],
                        grid=(N * 32 * _NSG, 1, 1),
                        threadgroup=(32 * _NSG, 1, 1),
                        output_shapes=[(M, N)],
                        output_dtypes=[mx.float16],
                        template=[("M", M), ("GROUP", 64), ("NSG", _NSG)],
                    )[0]
                    if (M, K, N) not in _hit_logged:
                        _hit_logged.add((M, K, N))
                        logger.info(
                            "fast_qmv active: M=%d K=%d N=%d (skinny path)",
                            M,
                            K,
                            N,
                        )
                    result = out.astype(x.dtype)
                    _stats["fast"] += 1
                    if _stats["fast"] % 500 == 0:
                        logger.info(
                            "fast_qmv stats: fast=%d fallback=%d",
                            _stats["fast"],
                            _stats["fallback"],
                        )
                    return result.reshape(x.shape[:-1] + (N,))
    except Exception as e:
        logger.warning("fast_qmv fallback: %s", e)
    _stats["fallback"] += 1
    return _orig_call(self, x)


def enable():
    global _orig_call
    if _orig_call is not None:
        return
    if not mx.metal.is_available():
        return
    _orig_call = nn.QuantizedLinear.__call__
    nn.QuantizedLinear.__call__ = _fast_call
    logger.info("fast_qmv enabled (skinny-M affine 4-bit fast path)")


def maybe_enable_from_env():
    if os.environ.get("MLX_VLM_FAST_QMV", "0") == "1":
        enable()
