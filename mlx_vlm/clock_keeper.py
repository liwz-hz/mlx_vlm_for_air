"""DVFS clock keeper for the speculative verify path.

The GPU governor downclocks under sustained memory-bound load (the verify
GEMV kernels at T=4 run at 17-39 GB/s depending on DVFS state). A tiny
ALU-only kernel interleaved into the stream keeps compute-utilization
signals high, holding clocks up: median +18% GEMV bandwidth in isolated
A/B. Enabled via MLX_VLM_CLOCK_KEEPER (default "1"); MLX_VLM_KEEPER_PERIOD
sets the per-projection tick period (default 8).
"""

import logging
import os

import mlx.core as mx

logger = logging.getLogger(__name__)

_SOURCE = """
    float acc = float(thread_position_in_grid.x) * 1e-6f;
    float4 v = float4(acc, acc + 1.0f, acc + 2.0f, acc + 3.0f);
    for (int i = 0; i < 2000; i++) {
      v = fma(v, float4(1.0000001f), v);
      v = fma(v, float4(0.9999999f), v);
    }
    if (v.x == 12345.678f) {
      out[0] = v.x;
    }
"""

_kernel = None
_tick = 0
_dummy_out = None


def _enabled():
    return os.environ.get("MLX_VLM_CLOCK_KEEPER", "0") == "1"


def _period():
    try:
        return max(1, int(os.environ.get("MLX_VLM_KEEPER_PERIOD", "8")))
    except ValueError:
        return 8


def _get_kernel():
    global _kernel
    if _kernel is None:
        _kernel = mx.fast.metal_kernel(
            name="clock_keeper_v1",
            input_names=[],
            output_names=["out"],
            source=_SOURCE,
        )
    return _kernel


def tick():
    """Lazily launch the keeper; returns a lazy scalar or None.

    The caller folds it into its output graph (out + k * 0) so the keeper
    executes in-stream without any eval/sync boundary.
    """
    global _tick
    if not _enabled() or not mx.metal.is_available():
        return None
    _tick += 1
    if _tick % _period() != 0:
        return None
    try:
        return _get_kernel()(
            inputs=[],
            grid=(128, 1, 1),
            threadgroup=(32, 1, 1),
            output_shapes=[(1,)],
            output_dtypes=[mx.float32],
        )[0]
    except Exception as e:
        logger.warning("clock_keeper failed: %s", e)
        return None
