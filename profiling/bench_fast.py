"""Compare the fast backend against the general one on identical trees.

Builds the same tree with the same seed for both backends, checks that the
outputs agree, and reports single-worker throughput for each.  Run it on the
target machine before trusting the numbers in the README — they were measured
on four cores and scale with core count and memory bandwidth.

    python profiling/bench_fast.py -W 540 -H 960 -n 12
"""

import argparse
import time

import numpy as np

from hecomes.artgen.fast import compile_fast, eval_fast, restrict_weights
from hecomes.artgen.tree import build_node, compile_plan, eval_plan, linearize
from hecomes.config import PERSONALITIES_DIR, load_personality_list


def time_eval(fn, plan, steps, repeats=3):
    fn(plan, steps)  # warm up
    t0 = time.perf_counter()
    for _ in range(repeats):
        result = fn(plan, steps)
    return len(steps) * repeats / (time.perf_counter() - t0), result


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("-W", "--width", type=int, default=540)
    ap.add_argument("-H", "--height", type=int, default=960)
    ap.add_argument("-n", "--n-trees", type=int, default=8)
    ap.add_argument("-c", "--chunk-size", type=int, default=10)
    ap.add_argument("-f", "--fps", type=int, default=30)
    ap.add_argument("--min-depth", type=int, default=6)
    ap.add_argument("--max-depth", type=int, default=12)
    ap.add_argument("--personality", default="personality")
    ap.add_argument("--skip-reference", action="store_true",
                    help="Only time the fast backend (the general one is slow at large sizes).")
    args = ap.parse_args()

    weights, dropped = restrict_weights(
        load_personality_list(PERSONALITIES_DIR / (args.personality + ".json"))
    )
    if dropped:
        print(f"dropped (no fast path): {', '.join(dropped)}")

    steps = (np.arange(args.chunk_size) / args.fps).astype(np.float32).reshape(-1, 1, 1, 1)
    ref_all, bi_all, nn_all = [], [], []

    for seed in range(args.n_trees):
        np.random.seed(seed)
        nodes, leaves = {}, {}
        root_id = build_node(0, args.min_depth, args.max_depth,
                             args.width, args.height, weights, 4e-3, nodes, leaves)
        order = linearize(root_id, nodes)
        chain = " -> ".join(nodes[nid].func.func.__name__ for nid in order)

        bi, out_bi = time_eval(eval_fast, compile_fast(order, nodes, leaves,
                                                       args.width, args.height, True), steps)
        nn, _ = time_eval(eval_fast, compile_fast(order, nodes, leaves,
                                                  args.width, args.height, False), steps)
        bi_all.append(bi)
        nn_all.append(nn)

        if args.skip_reference:
            print(f"seed {seed}: fast-bi {bi:6.1f} | fast-nn {nn:6.1f} fps   {chain}")
            continue

        ref, out_ref = time_eval(eval_plan, compile_plan(order, nodes, leaves), steps)
        ref_all.append(ref)
        diff = np.abs(out_ref - out_bi).max()
        print(f"seed {seed}: ref {ref:6.1f} | fast-bi {bi:6.1f} ({bi / ref:4.1f}x) | "
              f"fast-nn {nn:6.1f} ({nn / ref:4.1f}x) | maxdiff {diff:.1e}   {chain}")

    print(f"\n{args.width}x{args.height}, one worker, mean over {args.n_trees} trees:")
    if ref_all:
        print(f"  reference     {np.mean(ref_all):6.1f} fps")
    print(f"  fast bilinear {np.mean(bi_all):6.1f} fps")
    print(f"  fast nearest  {np.mean(nn_all):6.1f} fps")


if __name__ == "__main__":
    main()
