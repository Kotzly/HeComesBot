"""
Benchmark leaf evaluation: numpy (CPU) vs CuPy (GPU).
Leaf op: out = base + delta * steps  broadcast over chunk dimension.
"""
import time

import numpy as np
import cupy as cp

# ── Config ───────────────────────────────────────────────────────────────────
DX, DY     = 1024, 768
CHUNK_SIZE = 5
N_RUNS     = 50
SHAPE_BASE = (DY, DX, 3)
SHAPE_OUT  = (CHUNK_SIZE, DY, DX, 3)


def bench_cpu(base, delta, steps, n=N_RUNS):
    # warmup
    _ = base + delta * steps
    t0 = time.perf_counter()
    for _ in range(n):
        out = base + delta * steps
    return (time.perf_counter() - t0) / n * 1000


def bench_gpu(base_gpu, delta, steps_gpu, n=N_RUNS):
    # warmup + sync
    _ = base_gpu + delta * steps_gpu
    cp.cuda.Stream.null.synchronize()
    t0 = time.perf_counter()
    for _ in range(n):
        out = base_gpu + delta * steps_gpu
    cp.cuda.Stream.null.synchronize()
    return (time.perf_counter() - t0) / n * 1000


def bench_gpu_transfer(base, delta, steps, n=N_RUNS):
    """Includes CPU->GPU transfer and GPU->CPU transfer (realistic case)."""
    # warmup
    base_gpu  = cp.asarray(base)
    steps_gpu = cp.asarray(steps)
    out_gpu   = base_gpu + delta * steps_gpu
    _ = cp.asnumpy(out_gpu)
    cp.cuda.Stream.null.synchronize()

    t0 = time.perf_counter()
    for _ in range(n):
        base_gpu  = cp.asarray(base)
        steps_gpu = cp.asarray(steps)
        out_gpu   = base_gpu + delta * steps_gpu
        _         = cp.asnumpy(out_gpu)
    cp.cuda.Stream.null.synchronize()
    return (time.perf_counter() - t0) / n * 1000


if __name__ == "__main__":
    rng   = np.random.default_rng(42)
    base  = rng.standard_normal(SHAPE_BASE).astype(np.float32)
    steps = (np.arange(CHUNK_SIZE) / 30.0).astype(np.float32).reshape(-1, 1, 1, 1)
    delta = np.float32(4e-3)

    base_gpu  = cp.asarray(base)
    steps_gpu = cp.asarray(steps)

    # correctness
    ref     = base + delta * steps
    gpu_out = cp.asnumpy(base_gpu + delta * steps_gpu)
    match   = np.allclose(ref, gpu_out, atol=1e-6)

    ms_cpu      = bench_cpu(base, delta, steps)
    ms_gpu      = bench_gpu(base_gpu, delta, steps_gpu)
    ms_gpu_full = bench_gpu_transfer(base, delta, steps)

    print(f"\nShape base: {SHAPE_BASE}  chunk_size: {CHUNK_SIZE}  runs: {N_RUNS}\n")
    header = f"{'Implementation':<30} {'ms/call':>10} {'speedup':>10}   Correct?"
    sep = "-" * len(header)
    print(sep)
    print(header)
    print(sep)
    print(f"{'numpy (CPU)':<30} {ms_cpu:>10.3f} {'1.00x':>10}   ok")
    print(f"{'cupy (GPU compute only)':<30} {ms_gpu:>10.3f} {ms_cpu/ms_gpu:>9.2f}x   {'ok' if match else 'FAIL'}")
    print(f"{'cupy (GPU + transfers)':<30} {ms_gpu_full:>10.3f} {ms_cpu/ms_gpu_full:>9.2f}x   {'ok' if match else 'FAIL'}")
    print(sep)
    print("\nNote: 'GPU compute only' assumes arrays already live on GPU.")
    print("      'GPU + transfers' includes CPU->GPU and GPU->CPU per call.")
