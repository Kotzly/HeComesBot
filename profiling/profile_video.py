"""
Per-function timing analysis of eval_plan for a 1024x768 video workload.

Accumulates call count and wall time per function name, then prints a report
sorted by total time.
"""
import argparse
import time
from collections import defaultdict

import numpy as np
from tqdm import tqdm

from hecomes.artgen.tree import build_node, compile_plan, linearize
from hecomes.config import PERSONALITIES_DIR, load_personality_list

# ── Config ──────────────────────────────────────────────────────────────────
DX, DY   = 1024, 768
SEED     = 42
MIN_D, MAX_D = 6, 12
ALPHA    = 4e-3
N_CHUNKS = 5
CHUNK_SZ = 5
FPS      = 30
PERSONALITY = "personality"


# ── Timed eval_plan ──────────────────────────────────────────────────────────
def eval_plan_timed(plan: list, steps, timings: dict, counts: dict, use_gpu: bool = False) -> np.ndarray:
    if use_gpu:
        import cupy as cp
        steps = cp.asarray(steps)

    buf = [None] * len(plan)
    for i, (func, params, base, delta, children) in enumerate(plan):
        t0 = time.perf_counter()
        if func is None:
            buf[i] = (cp.asarray(base) if use_gpu else base) + delta * steps
            name = "leaf"
        else:
            args = [buf[j] for j in children]
            for j in children:
                buf[j] = None
            buf[i] = func(*args, **params)
            name = func.__name__
        if use_gpu:
            cp.cuda.Stream.null.synchronize()
        elapsed = time.perf_counter() - t0
        timings[name] += elapsed
        counts[name] += 1

    result = buf[-1]
    if use_gpu:
        return cp.asnumpy(result)
    return result


# ── Profiling target ─────────────────────────────────────────────────────────
def run(use_gpu: bool = False):
    p = load_personality_list(PERSONALITIES_DIR / (PERSONALITY + ".json"))

    np.random.seed(SEED % (2**32 - 1))
    nodes, leaves = {}, {}
    root_id = build_node(0, MIN_D, MAX_D, DX, DY, p, ALPHA, nodes, leaves)
    order = linearize(root_id, nodes)
    plan = compile_plan(order, nodes, leaves)

    all_steps = (np.arange(N_CHUNKS * CHUNK_SZ) / FPS).astype(np.float32).reshape(-1, 1, 1, 1)
    chunk_steps = [
        all_steps[s : s + CHUNK_SZ] for s in range(0, N_CHUNKS * CHUNK_SZ, CHUNK_SZ)
    ]

    timings = defaultdict(float)
    counts  = defaultdict(int)

    for chunk in tqdm(chunk_steps):
        raw = eval_plan_timed(plan, chunk, timings, counts, use_gpu=use_gpu)
        np.rint(raw.clip(0, 1) * 255.0).astype(np.uint8)

    return timings, counts


# ── Report ────────────────────────────────────────────────────────────────────
def print_report(timings, counts):
    total = sum(timings.values())
    rows = sorted(timings.items(), key=lambda x: -x[1])

    header = f"{'Function':<30} {'Calls':>8} {'Total(s)':>10} {'Avg(ms)':>10} {'%Total':>8}"
    sep = "-" * len(header)
    lines = [sep, header, sep]
    for name, t in rows:
        n = counts[name]
        avg_ms = (t / n) * 1000
        pct = t / total * 100
        lines.append(f"{name:<30} {n:>8} {t:>10.4f} {avg_ms:>10.4f} {pct:>7.2f}%")
    lines.append(sep)
    lines.append(f"{'TOTAL':<30} {sum(counts.values()):>8} {total:>10.4f}")
    lines.append(sep)
    report = "\n".join(lines)
    print(report)
    return report


# ── Main ─────────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Per-function timing of eval_plan.")
    parser.add_argument("--gpu", action="store_true", default=False,
                        help="Run eval_plan on GPU via CuPy. Default: CPU.")
    cli_args = parser.parse_args()

    timings, counts = run(use_gpu=cli_args.gpu)
    report = print_report(timings, counts)
    with open("profile_report.txt", "w") as f:
        f.write(report + "\n")
    print("\nFull report written to profile_report.txt")
