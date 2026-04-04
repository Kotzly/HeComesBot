"""
Benchmark Numba-jitted leaf evaluation vs current numpy approach.
Leaf op: out[t] = base + delta * steps[t]  for each frame t in chunk.
"""
import time

import numpy as np
import numba

# ── Config ───────────────────────────────────────────────────────────────────
DX, DY     = 1024, 768
CHUNK_SIZE = 5
N_RUNS     = 50
SHAPE_BASE = (DY, DX, 3)
SHAPE_OUT  = (CHUNK_SIZE, DY, DX, 3)


# ── Implementations ───────────────────────────────────────────────────────────

def numpy_leaf(base, delta, steps):
    """Current: numpy broadcast add — materialises chunk_sz copies of base."""
    return base + delta * steps


@numba.njit(parallel=True, cache=True)
def _numba_leaf_frames(base, delta, steps_flat, out):
    """Parallel over frames (chunk_sz iterations)."""
    for t in numba.prange(steps_flat.shape[0]):
        out[t] = base + delta * steps_flat[t]


@numba.njit(parallel=True, cache=True)
def _numba_leaf_pixels(base, delta, steps_flat, out):
    """Parallel over pixels (H*W iterations), loop frames in inner loop."""
    n_frames = steps_flat.shape[0]
    H, W, C = base.shape
    for p in numba.prange(H * W):
        y = p // W
        x = p % W
        for t in range(n_frames):
            s = delta * steps_flat[t]
            for c in range(C):
                out[t, y, x, c] = base[y, x, c] + s


def numba_leaf_frames(base, delta, steps, out):
    _numba_leaf_frames(base, delta, steps[:, 0, 0, 0], out)
    return out


def numba_leaf_pixels(base, delta, steps, out):
    _numba_leaf_pixels(base, delta, steps[:, 0, 0, 0], out)
    return out


# ── Benchmark ─────────────────────────────────────────────────────────────────
def bench(name, fn, *args, n=N_RUNS):
    fn(*args)  # warmup
    t0 = time.perf_counter()
    for _ in range(n):
        fn(*args)
    return (time.perf_counter() - t0) / n * 1000


if __name__ == "__main__":
    rng = np.random.default_rng(42)
    base  = rng.standard_normal(SHAPE_BASE).astype(np.float32)
    steps = (np.arange(CHUNK_SIZE) / 30.0).astype(np.float32).reshape(-1, 1, 1, 1)
    delta = np.float32(4e-3)
    out   = np.empty(SHAPE_OUT, dtype=np.float32)

    print("Compiling Numba kernels...", end=" ", flush=True)
    numba_leaf_frames(base, delta, steps, out)
    numba_leaf_pixels(base, delta, steps, out)
    print("done.")

    ref = numpy_leaf(base, delta, steps)

    impls = [
        ("numpy broadcast",      lambda: numpy_leaf(base, delta, steps)),
        ("numba parallel/frames",lambda: numba_leaf_frames(base, delta, steps, out)),
        ("numba parallel/pixels",lambda: numba_leaf_pixels(base, delta, steps, out)),
    ]

    print(f"\nShape base: {SHAPE_BASE}  chunk_size: {CHUNK_SIZE}  runs: {N_RUNS}\n")
    header = f"{'Implementation':<28} {'ms/call':>10} {'speedup':>10}   Correct?"
    sep = "-" * len(header)
    print(sep)
    print(header)
    print(sep)

    baseline = None
    for name, fn in impls:
        ms = bench(name, fn)
        if baseline is None:
            baseline = ms
        result = fn()
        match = np.allclose(ref, result if not isinstance(result, type(None)) else out, atol=1e-6)
        print(f"{name:<28} {ms:>10.3f} {baseline/ms:>9.2f}x   {'ok' if match else 'FAIL'}")

    print(sep)
