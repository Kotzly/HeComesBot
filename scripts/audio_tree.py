"""Minimal CLI wrapper around :mod:`hecomes.audiogen`.

Run:
    python scripts/audio_tree.py --seconds 4 --out out.wav
"""

from __future__ import annotations

import argparse

from hecomes.audiogen import SAMPLE_RATE, generate_wav


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--seconds", type=float, default=3.0)
    p.add_argument("--out", type=str, default="audio_tree.wav")
    p.add_argument("--seed", type=int, default=None)
    p.add_argument("--min-depth", type=int, default=6)
    p.add_argument("--max-depth", type=int, default=10)
    args = p.parse_args()

    generate_wav(
        args.out,
        seconds=args.seconds,
        seed=args.seed,
        min_depth=args.min_depth,
        max_depth=args.max_depth,
        verbose=True,
    )
    print(f"wrote {args.out}  ({args.seconds}s @ {SAMPLE_RATE} Hz)")


if __name__ == "__main__":
    main()
