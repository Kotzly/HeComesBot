"""Check that the fast paths still agree with the reference implementations.

Three invariants, none of which needs a second checkout:

1. ``separable_blur`` / ``separable_sharpen`` against the dense 5x5 kernels
   they replaced (kept in ``func_utils`` for exactly this purpose).
2. The compiled resampler against ``scipy.ndimage.map_coordinates``, warp by
   warp — the same per-channel call the operators used to make.
3. The general plan against the fast plan on whole trees, which must be
   bit-identical with bilinear sampling since both share ``resample``.

Run at a few sizes, 4K included, since the warp coordinate error grows with
resolution:

    python profiling/verify_parity.py --sizes 256x256 512x512 3840x2160
"""

import argparse
import sys

import numpy as np
from scipy.ndimage import map_coordinates

from hecomes.artgen.fast import compile_fast, eval_fast, restrict_weights
from hecomes.artgen.func_utils import (
    _apply_kernel,
    _gaussian_kernel_5,
    _sharpen_kernel_5,
    separable_blur,
    separable_sharpen,
)
from hecomes.artgen.resample import WARP_MAPS, apply_resampler, compile_resampler
from hecomes.artgen.tree import build_node, compile_plan, eval_plan, linearize
from hecomes.config import PERSONALITIES_DIR, load_personality_list

# Tolerances are on [0, 1] pixel values. 1/255 is one step of 8-bit output.
TOL_EXACT = 1e-5
TOL_WARP = 1.0 / 255

WARP_PARAMS = {
    "swirl": dict(cx=0.13, cy=-0.21, strength=1.3, power=-2.0),
    "ripple": dict(ax=0.12, ay=0.08, kx=3.0, ky=5.0, phase_x=0.3, phase_y=1.1),
    "pinch": dict(cx=0.11, cy=0.09, strength=0.5),
    "polar_warp": dict(cx=0.1, cy=-0.05),
}


def smooth_image(w, h):
    """Smooth content, which is what a tree actually feeds a warp.

    White noise is not a fair test here: any sub-pixel shift in the sampling
    position swaps in an unrelated value, so it reports a large difference for
    a change that is invisible on real content.
    """
    xs = np.linspace(-1, 1, w, dtype=np.float32)[None, :]
    ys = np.linspace(-1, 1, h, dtype=np.float32)[:, None]
    img = np.stack([
        np.sin(3 * xs + 2 * ys) * 0.5 + 0.5,
        np.cos(4 * np.sqrt(xs**2 + ys**2)) * 0.5 + 0.5,
        (xs * ys) * 0.5 + 0.5,
    ], axis=-1)
    return np.ascontiguousarray(np.broadcast_to(img.astype(np.float32), (h, w, 3)))[None]


def report(label, diff, tol):
    worst = diff.max()
    over = (diff.max(axis=-1) > tol).mean() * 100 if diff.ndim == 4 else 0.0
    ok = worst <= tol
    print(f"  {'ok ' if ok else 'FAIL'} {label:34s} max {worst:.2e}  px over tol {over:.5f}%")
    return ok


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--sizes", nargs="+", default=["256x256", "512x512"],
                    help="WxH sizes to check. Add 3840x2160 for 4K.")
    ap.add_argument("-n", "--n-trees", type=int, default=6)
    args = ap.parse_args()

    weights, _ = restrict_weights(
        load_personality_list(PERSONALITIES_DIR / "personality.json")
    )
    passed = True

    for size in args.sizes:
        w, h = (int(v) for v in size.split("x"))
        print(f"\n=== {w}x{h} ===")
        img = smooth_image(w, h)

        print(" separable kernels vs the dense 5x5 form")
        passed &= report("blur", np.abs(separable_blur(img.copy()) - _apply_kernel(img[0], _gaussian_kernel_5)), TOL_EXACT)
        passed &= report("sharpen", np.abs(separable_sharpen(img.copy()) - _apply_kernel(img[0], _sharpen_kernel_5)), TOL_EXACT)

        print(" compiled resampler vs map_coordinates, per warp")
        for name, params in WARP_PARAMS.items():
            x_src, y_src = WARP_MAPS[name](w, h, **params)
            got = apply_resampler(compile_resampler(x_src, y_src, w, h, True), img.copy())
            col = (x_src + 1) / 2 * (w - 1)
            row = (y_src + 1) / 2 * (h - 1)
            expected = np.stack(
                [map_coordinates(img[0][:, :, c], [row, col], order=1, mode="nearest")
                 for c in range(3)], axis=-1,
            )[None]
            passed &= report(name, np.abs(got - expected), TOL_WARP)

        print(" general plan vs fast plan, whole trees")
        worst = 0.0
        for seed in range(args.n_trees):
            np.random.seed(seed)
            nodes, leaves = {}, {}
            root_id = build_node(0, 6, 12, w, h, weights, 4e-3, nodes, leaves)
            order = linearize(root_id, nodes)
            steps = np.zeros((1, 1, 1, 1), np.float32)
            general = eval_plan(compile_plan(order, nodes, leaves), steps)
            fast = eval_fast(compile_fast(order, nodes, leaves, w, h, True), steps)
            worst = max(worst, np.abs(general - fast).max())
        passed &= report(f"{args.n_trees} trees", np.array(worst), TOL_EXACT)

    print("\nPASS" if passed else "\nFAIL")
    return 0 if passed else 1


if __name__ == "__main__":
    sys.exit(main())
