"""
Benchmark eval_plan across different chunk sizes for a fixed total number of frames.
Measures wall time for leaf eval and total eval_plan, keeping total frames constant.
"""
import time
from collections import defaultdict

import numpy as np
from tqdm import tqdm

from hecomes.artgen.tree import build_node, compile_plan, linearize
from hecomes.config import PERSONALITIES_DIR, load_personality_list

# ── Config ───────────────────────────────────────────────────────────────────
DX, DY       = 1024, 768
SEED         = 42
MIN_D, MAX_D = 6, 12
ALPHA        = 4e-3
PERSONALITY  = "personality"
FPS          = 30
N_FRAMES     = 60          # fixed total — chunk size divides this
CHUNK_SIZES  = [1, 2, 3, 4, 5, 6, 10, 12, 15, 20]


# ── Timed eval_plan ──────────────────────────────────────────────────────────
def eval_plan_timed(plan, steps, timings, counts):
    buf = [None] * len(plan)
    for i, (func, params, base, delta, children) in enumerate(plan):
        t0 = time.perf_counter()
        if func is None:
            buf[i] = base + delta * steps
            name = "leaf"
        else:
            args = [buf[j] for j in children]
            for j in children:
                buf[j] = None
            buf[i] = func(*args, **params)
            name = func.__name__
        timings[name] += time.perf_counter() - t0
        counts[name] += 1
    return buf[-1]


# ── Build tree once ───────────────────────────────────────────────────────────
def build(p):
    np.random.seed(SEED % (2**32 - 1))
    nodes, leaves = {}, {}
    root_id = build_node(0, MIN_D, MAX_D, DX, DY, p, ALPHA, nodes, leaves)
    order = linearize(root_id, nodes)
    return compile_plan(order, nodes, leaves)


# ── Run one chunk-size config ─────────────────────────────────────────────────
def run(plan, chunk_size):
    all_steps = (np.arange(N_FRAMES) / FPS).astype(np.float32).reshape(-1, 1, 1, 1)
    chunks = [all_steps[s:s + chunk_size] for s in range(0, N_FRAMES, chunk_size)]

    timings = defaultdict(float)
    counts  = defaultdict(int)

    t_total = time.perf_counter()
    for chunk in chunks:
        raw = eval_plan_timed(plan, chunk, timings, counts)
        np.rint(raw.clip(0, 1) * 255.0).astype(np.uint8)
    t_total = time.perf_counter() - t_total

    return t_total, timings, counts


# ── Main ──────────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    p = load_personality_list(PERSONALITIES_DIR / (PERSONALITY + ".json"))
    plan = build(p)

    print(f"\nResolution: {DX}x{DY}  |  Total frames: {N_FRAMES}  |  Seed: {SEED}\n")
    header = f"{'chunk_size':>12} {'n_chunks':>10} {'total(s)':>10} {'leaf(s)':>10} {'leaf%':>8} {'ms/frame':>10}"
    sep = "-" * len(header)
    print(sep)
    print(header)
    print(sep)

    for chunk_size in tqdm(CHUNK_SIZES):
        t_total, timings, counts = run(plan, chunk_size)
        n_chunks = N_FRAMES // chunk_size
        leaf_t = timings["leaf"]
        leaf_pct = leaf_t / sum(timings.values()) * 100
        ms_per_frame = t_total / N_FRAMES * 1000
        print(f"{chunk_size:>12} {n_chunks:>10} {t_total:>10.3f} {leaf_t:>10.3f} {leaf_pct:>7.1f}% {ms_per_frame:>10.1f}")

    print(sep)
