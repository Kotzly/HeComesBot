"""
Benchmark full eval_plan: CPU (numpy) vs GPU (CuPy).
Uses the same tree and chunk config as profile_video.py.
"""
import time

import numpy as np

from hecomes.artgen.tree import build_node, compile_plan, eval_plan, linearize
from hecomes.config import PERSONALITIES_DIR, load_personality_list

# ── Config ───────────────────────────────────────────────────────────────────
DX, DY       = 1024, 768
SEED         = 42
MIN_D, MAX_D = 6, 12
ALPHA        = 4e-3
PERSONALITY  = "personality"
FPS          = 30
CHUNK_SIZE   = 5
N_RUNS       = 10


def bench(plan, steps, use_gpu=False, n=N_RUNS):
    eval_plan(plan, steps, use_gpu=use_gpu)   # warmup
    t0 = time.perf_counter()
    for _ in range(n):
        eval_plan(plan, steps, use_gpu=use_gpu)
    return (time.perf_counter() - t0) / n * 1000


if __name__ == "__main__":
    p = load_personality_list(PERSONALITIES_DIR / (PERSONALITY + ".json"))

    np.random.seed(SEED % (2**32 - 1))
    nodes, leaves = {}, {}
    root_id = build_node(0, MIN_D, MAX_D, DX, DY, p, ALPHA, nodes, leaves)
    order = linearize(root_id, nodes)

    plan = compile_plan(order, nodes, leaves)

    steps = (np.arange(CHUNK_SIZE) / FPS).astype(np.float32).reshape(-1, 1, 1, 1)

    # correctness check — deep trees near safe_divide singularities cause
    # float32 rounding differences to amplify; check both are finite and
    # mean difference is small rather than max (a few chaotic pixels are ok)
    out_cpu = eval_plan(plan, steps, use_gpu=False)
    out_gpu = eval_plan(plan, steps, use_gpu=True)
    both_finite = np.all(np.isfinite(out_cpu)) and np.all(np.isfinite(out_gpu))
    mean_close = np.mean(np.abs(out_cpu - out_gpu)) < 1e-2
    match = both_finite and mean_close

    ms_cpu = bench(plan, steps, use_gpu=False)
    ms_gpu = bench(plan, steps, use_gpu=True)

    print(f"\nResolution: {DX}x{DY}  chunk_size: {CHUNK_SIZE}  runs: {N_RUNS}\n")
    header = f"{'Implementation':<25} {'ms/call':>10} {'speedup':>10}   Correct?"
    sep = "-" * len(header)
    print(sep)
    print(header)
    print(sep)
    print(f"{'eval_plan CPU':<25} {ms_cpu:>10.1f} {'1.00x':>10}   ok")
    print(f"{'eval_plan GPU':<25} {ms_gpu:>10.1f} {ms_cpu/ms_gpu:>9.2f}x   {'ok' if match else 'FAIL'}")
    print(sep)
