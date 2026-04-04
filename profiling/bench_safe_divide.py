"""
Benchmark competing safe_divide implementations on realistic array sizes.
Arrays match the video workload: (chunk_sz, H, W, C) = (5, 768, 1024, 3).
"""
import time
import numpy as np

SHAPES = [
    (5, 256,  256,  3),
    (5, 768,  1024, 3),
    (5, 1080, 1920, 3),
]
EPS = np.float32(1e-3)
N_RUNS = 20


def make_inputs(shape, seed=0):
    rng = np.random.default_rng(seed)
    a = rng.standard_normal(shape).astype(np.float32)
    b = rng.standard_normal(shape).astype(np.float32)
    # ensure some values are near zero to exercise the guard
    b.ravel()[::50] = 0.0
    return a, b


def bench(name, fn, a, b, n=N_RUNS):
    # warmup
    fn(a, b)
    t0 = time.perf_counter()
    for _ in range(n):
        fn(a, b)
    elapsed = time.perf_counter() - t0
    return elapsed / n * 1000  # ms per call


# ── Implementations ───────────────────────────────────────────────────────────

def current(a, b, eps=EPS):
    """Current: abs + where."""
    return a / np.where(np.abs(b) < eps, eps, b)


def sign_maximum(a, b, eps=EPS):
    """sign * maximum(abs, eps) — avoids where, two ops on b."""
    safe_b = np.sign(b) * np.maximum(np.abs(b), eps)
    return a / safe_b


def clip_abs(a, b, eps=EPS):
    """Clip abs(b) then restore sign via copysign."""
    safe_b = np.copysign(np.maximum(np.abs(b), eps), b)
    return a / safe_b


def errstate(a, b, eps=EPS):
    """Divide ignoring errors, replace ±inf with ±1/eps, nan with 0."""
    with np.errstate(divide='ignore', invalid='ignore'):
        out = a / b
    inf_mask = np.isinf(out)
    nan_mask = np.isnan(out)
    out[inf_mask] = np.sign(out[inf_mask]) * np.float32(1 / eps)
    out[nan_mask] = np.float32(0.0)
    return out


def inplace_clamp(a, b, eps=EPS):
    """Copy b, clamp in-place, divide."""
    safe_b = b.copy()
    mask = np.abs(safe_b) < eps
    safe_b[mask] = eps
    return a / safe_b


def maximum_only(a, b, eps=EPS):
    """np.maximum on abs(b), no sign restore — only correct when b>=0."""
    # included for reference / lower-bound timing
    return a / np.maximum(np.abs(b), eps)


# ── Run ────────────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    impls = [
        ("current (where+abs)",   current),
        ("sign * maximum(abs,e)", sign_maximum),
        ("copysign + maximum",    clip_abs),
        ("errstate + isfinite",   errstate),
        ("inplace clamp",         inplace_clamp),
        ("maximum(abs,e) only",   maximum_only),
    ]

    header = f"{'Implementation':<30} {'ms/call':>10} {'speedup':>10}   Correct?"
    sep = "-" * len(header)

    for shape in SHAPES:
        a, b = make_inputs(shape)
        ref = current(a, b)

        print(f"\nArray shape: {shape}  |  {N_RUNS} runs each\n")
        print(sep)
        print(header)
        print(sep)

        baseline = None
        for name, fn in impls:
            ms = bench(name, fn, a, b)
            if baseline is None:
                baseline = ms
            speedup = baseline / ms
            result = fn(a, b)
            correct = result.shape == ref.shape and np.all(np.isfinite(result))
            print(f"{name:<30} {ms:>10.2f} {speedup:>9.2f}x   {'ok' if correct else 'FAIL'}")

        print(sep)
