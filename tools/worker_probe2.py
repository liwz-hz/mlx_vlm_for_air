"""Cross-process CPU/GPU interference probe v2.

Both sides run for a fixed window, synchronized via pipe handshake.
Phases alternate within one session to cancel thermal drift:
  solo-cpu -> concurrent -> solo-cpu -> concurrent -> solo-cpu
The parent (GPU) prints its rate for each concurrent window; the child
prints per-phase CPU rates. Compare first/last solo vs concurrent.

Usage: python3 worker_probe2.py            # parent, spawns child itself
"""
import ctypes
import subprocess
import sys
import time

import numpy as np

M, K, NC, N_DOWN = 2048, 5120, 2176, 5120
WINDOW = 6.0  # seconds per phase


def child_main(read_fd, write_fd):
    import pathlib

    r = os.fdopen(read_fd, "rb")
    w = os.fdopen(write_fd, "wb")

    lib = ctypes.CDLL(str(pathlib.Path("mlx_vlm/libcpu_share.dylib").resolve()))
    u16p = ctypes.POINTER(ctypes.c_uint16)
    f32p = ctypes.POINTER(ctypes.c_float)
    lib.mlp_cpu_share.restype = ctypes.c_int
    lib.mlp_cpu_share.argtypes = [u16p, u16p, u16p, f32p, f32p] + [ctypes.c_size_t] * 4

    rng = np.random.default_rng(0)
    x = rng.integers(0, 65535, (M, K), dtype=np.uint16)
    wg = rng.integers(0, 65535, (NC, K), dtype=np.uint16)
    wu = rng.integers(0, 65535, (NC, K), dtype=np.uint16)
    wd = rng.standard_normal((N_DOWN, NC)).astype(np.float32)
    out = np.zeros((M, N_DOWN), np.float32)

    def one():
        rc = lib.mlp_cpu_share(
            x.ctypes.data_as(u16p), wg.ctypes.data_as(u16p),
            wu.ctypes.data_as(u16p), wd.ctypes.data_as(f32p),
            out.ctypes.data_as(f32p), M, K, NC, N_DOWN,
        )
        assert rc == 0

    one()
    w.write(b"READY\n")
    w.flush()
    # phases: 0=solo 1=concurrent 2=solo 3=concurrent 4=solo
    for phase in range(5):
        r.read(1)  # wait for parent's go
        n = 0
        best = 1e9
        t_end = time.perf_counter() + WINDOW
        while time.perf_counter() < t_end:
            t0 = time.perf_counter()
            one()
            dt = time.perf_counter() - t0
            best = min(best, dt)
            n += 1
        w.write(f"{phase} {n} {best*1e3:.1f}\n".encode())
        w.flush()
    w.write(b"DONE\n")
    w.flush()


def parent_main():
    import mlx.core as mx

    rng = np.random.default_rng(1)
    w_ = mx.array(rng.standard_normal((17408, 5120)).astype(np.float32) * 0.02).astype(mx.bfloat16)
    wq, s, b = mx.quantize(w_, group_size=64, bits=4)
    x = mx.array(rng.standard_normal((M, K)).astype(np.float32) * 0.5).astype(mx.bfloat16)
    mx.eval(wq, s, b, x)

    def gpu_one():
        o = mx.quantized_matmul(x, wq, s, b, transpose=True, group_size=64, bits=4)
        mx.eval(o)

    gpu_one()

    r_pipe, w_child = os.pipe()
    r_parent, w_pipe = os.pipe()
    child = subprocess.Popen(
        [sys.executable, __file__, "--child", str(r_pipe), str(w_pipe)],
        pass_fds=(r_pipe, w_pipe),
    )
    rf = os.fdopen(r_parent, "rb")
    wf = os.fdopen(w_child, "wb")
    assert rf.readline() == b"READY\n"

    phases = [False, True, False, True, False]  # True = run GPU concurrently
    for phase, gpu_on in enumerate(phases):
        wf.write(b"g")
        wf.flush()
        n = 0
        best = 1e9
        t_end = time.perf_counter() + WINDOW
        while time.perf_counter() < t_end:
            t0 = time.perf_counter()
            gpu_one()
            dt = time.perf_counter() - t0
            best = min(best, dt)
            n += 1
        line = rf.readline().decode().strip()
        gpu_tag = "GPU+CPU" if gpu_on else "GPU-solo"
        print(f"phase{phase} {gpu_tag}: gpu_best={best*1e3:6.1f} ms   child: {line}", flush=True)
    print(rf.readline().decode().strip())
    child.wait()


if __name__ == "__main__":
    import os

    if len(sys.argv) > 1 and sys.argv[1] == "--child":
        child_main(int(sys.argv[2]), int(sys.argv[3]))
    else:
        parent_main()
