"""Real-time video generation CLI (``hecomes-video-fast``).

Uses the stripped-down backend in :mod:`hecomes.artgen.fast`: static warps are
pre-compiled to a gather, convolutions are separable, and elementwise ops run
in place.  Animation is the same uniform leaf drift as ``hecomes-video``.

Trees are drawn from the personality as usual, but only from the operators the
fast backend implements (see :data:`hecomes.artgen.fast.FAST_FUNCTIONS`).
Unless ``--no-budget`` is given, each candidate tree is timed on one real chunk
and redrawn if its throughput would fall short of the target frame rate, so the
generator is measured against the clock rather than assumed to beat it.
"""

import argparse
import multiprocessing as mp
import os
import time

import numpy as np

from hecomes.artgen.fast import compile_fast, eval_fast, restrict_weights
from hecomes.artgen.func_utils import hsv_to_rgb
from hecomes.artgen.tree import build_node, linearize
from hecomes.cli._video_utils import (
    build_ffmpeg_cmd,
    run_ffmpeg_pipeline,
    select_codec,
    setup_ffmpeg_path,
)
from hecomes.config import PERSONALITIES_DIR, load_personality_list

_plan = None
_color_space = "rgb"


def init_worker(plan, color_space):
    """Pool initialiser — plans are plain data, so this works under fork and spawn."""
    global _plan, _color_space
    _plan, _color_space = plan, color_space


def compute_chunk(steps):
    """Evaluate one chunk of frames and pack it as rgb24 bytes."""
    raw = eval_fast(_plan, steps)
    if _color_space == "hsv":
        frames = hsv_to_rgb(
            np.stack(
                [raw[..., 0] % 1.0, raw[..., 1].clip(0, 1), raw[..., 2].clip(0, 1)],
                axis=-1,
            )
        )
    elif _color_space == "cmy":
        frames = 1.0 - raw.clip(0, 1)
    else:
        frames = raw.clip(0, 1, out=raw)
    frames *= 255.0
    return np.rint(frames, out=frames).astype(np.uint8)


def build_plan(seed, weights, args):
    """Draw a tree for ``seed`` and compile it for the fast backend."""
    np.random.seed(seed % (2**32 - 1))
    nodes, leaves = {}, {}
    root_id = build_node(
        0, args.min_depth, args.max_depth, args.width, args.height,
        weights, args.drift, nodes, leaves,
    )
    order = linearize(root_id, nodes)
    plan = compile_fast(
        order, nodes, leaves, args.width, args.height,
        bilinear=(args.sampling == "bilinear"),
    )
    return plan, [nodes[nid].func.func.__name__ for nid in order]


#: How far generation must beat the target frame rate.  The encoder shares the
#: same CPU, so raw generation throughput overstates what comes out the far end
#: of the pipe.  Measured at 540x960 with libx264 ``-preset veryfast`` on four
#: cores: 32.5 fps of generation delivered 29.4 fps of finished video.
_ENCODER_HEADROOM = 1.15


def measure_fps(plan, color_space, chunks, processes):
    """Aggregate generation throughput of the real worker pool, in fps.

    Measured with the pool rather than by timing one worker and multiplying:
    the workers are memory-bandwidth bound, so they scale sublinearly (82% of
    linear at three workers on a four-core box).  Multiplying a solo timing
    overstates the pipeline by a fifth or more, which is exactly the margin
    the budget is meant to police.
    """
    n_frames = sum(len(c) for c in chunks)
    with mp.Pool(processes, initializer=init_worker, initargs=(plan, color_space)) as pool:
        for _ in pool.imap(compute_chunk, chunks):  # warm up workers and caches
            pass
        t0 = time.perf_counter()
        for _ in pool.imap(compute_chunk, chunks):
            pass
    return n_frames / (time.perf_counter() - t0)


def draw_plan_within_budget(seed, weights, args, chunks):
    """Draw trees until one clears the frame-rate budget.

    Returns ``(plan, chain, fps, seed)`` where ``fps`` is the measured
    aggregate generation throughput.  With ``--no-budget`` the first tree is
    returned whatever it measures.
    """
    target = (args.budget_fps or args.fps) * _ENCODER_HEADROOM
    for attempt in range(args.max_tries):
        plan, chain = build_plan(seed, weights, args)
        fps = measure_fps(plan, args.color_space, chunks, args.processes)
        if args.no_budget or fps >= target:
            if attempt:
                print(f"  ({attempt} tree(s) rejected as too slow)")
            return plan, chain, fps, seed
        print(f"  seed {seed}: {fps:.1f} fps < {target:.1f} needed "
              f"({' -> '.join(chain)}) — redrawing")
        seed = (seed * 6364136223846793005 + 1442695040888963407) % (2**32 - 1)
    raise SystemExit(
        f"No tree met the {target:.1f} fps budget in {args.max_tries} tries at "
        f"{args.width}x{args.height} with {args.processes} worker(s). "
        f"Try --sampling nearest, more --processes, a smaller frame, "
        f"or a lower --max-depth."
    )


def parse_args(argv=None):
    p = argparse.ArgumentParser(
        prog="hecomes-video-fast",
        description="Generate videos faster than real time.",
    )
    p.add_argument("-o", "--output", default=None,
                   help="Output path. Default: videos/fast-N.<ext>.")
    p.add_argument("-n", "--n-videos", type=int, default=1, help="Number of videos.")
    p.add_argument("-W", "--width", type=int, default=540, help="Frame width.")
    p.add_argument("-H", "--height", type=int, default=960, help="Frame height.")
    p.add_argument("-f", "--fps", type=int, default=30, help="Frames per second.")
    p.add_argument("-d", "--duration", type=float, default=10.0, help="Duration in seconds.")
    p.add_argument("-S", "--seed", type=int, default=None, help="Fixed seed (one video).")
    p.add_argument("-p", "--processes", type=int, default=max(1, (os.cpu_count() or 2) - 1),
                   help="Worker processes. Default: CPU count minus one.")
    p.add_argument("-c", "--chunk-size", type=int, default=10, help="Frames per batch.")
    p.add_argument("-e", "--extension", default="mp4", help="Container: mp4, webm, avi, gif, ...")
    p.add_argument("-b", "--bitrate", default="6M", help="Constant bitrate.")
    p.add_argument("-C", "--codec", default=None, help="Codec. Default: recommended for the container.")
    p.add_argument("--personality", default="personality", help="Personality name in data/personalities/.")
    p.add_argument("--color-space", choices=("rgb", "hsv", "cmy"), default="rgb")
    p.add_argument("--sampling", choices=("bilinear", "nearest"), default="bilinear",
                   help="Warp interpolation. 'nearest' is ~4x cheaper per warp "
                        "at the cost of visible aliasing. Default: bilinear.")
    p.add_argument("--min-depth", type=int, default=6, help="Minimum tree depth.")
    p.add_argument("--max-depth", type=int, default=12, help="Maximum tree depth.")
    p.add_argument("--drift", type=float, default=4e-3, help="Leaf drift per second.")
    p.add_argument("--budget-fps", type=float, default=None,
                   help="Throughput a tree must reach to be accepted. Default: --fps.")
    p.add_argument("--max-tries", type=int, default=20,
                   help="Trees to draw before giving up on the budget.")
    p.add_argument("--no-budget", action="store_true",
                   help="Accept the first tree drawn, however slow.")
    p.add_argument("--preset", default="veryfast",
                   help="Encoder speed preset for codecs that take one (x264/x265). "
                        "At 540x960 'medium' encodes ~57 fps and 'veryfast' ~139 fps, "
                        "so the default keeps the encoder off the critical path.")
    p.add_argument("--benchmark", action="store_true",
                   help="Measure throughput and exit without encoding.")
    return p.parse_args(argv)


#: H.264 encoders, which need explicit 4:2:0 chroma for portable output.
_H264_CODECS = ("libx264", "libx265", "libopenh264", "h264_nvenc")

#: Codecs whose ffmpeg encoder accepts ``-preset``.
_PRESET_CODECS = ("libx264", "libx265")

#: Chroma subsampling every player understands. Without it ffmpeg picks
#: yuv444p for rgb24 input, which most browsers and Instagram refuse to play.
_COMPAT_PIX_FMT = "yuv420p"


def main(argv=None):
    setup_ffmpeg_path()
    args = parse_args(argv)

    weights, dropped = restrict_weights(
        load_personality_list(PERSONALITIES_DIR / (args.personality + ".json"))
    )
    if dropped:
        print(f"Note: no fast path for {', '.join(dropped)} — dropped from the personality.")

    codec = select_codec(args.extension, args.codec)
    if codec in _H264_CODECS and (args.width % 2 or args.height % 2):
        raise SystemExit(
            f"{args.width}x{args.height}: H.264 4:2:0 needs even dimensions. "
            f"Round both up to the next even number."
        )

    n_frames = int(round(args.fps * args.duration))
    all_steps = (np.arange(n_frames) / args.fps).astype(np.float32).reshape(-1, 1, 1, 1)
    chunk_steps = [all_steps[s:s + args.chunk_size] for s in range(0, n_frames, args.chunk_size)]
    # One chunk per worker: enough to see contention, cheap enough to redraw on.
    probe_chunks = chunk_steps[:args.processes] or chunk_steps

    n_videos = 1 if args.seed is not None else args.n_videos
    if args.output and n_videos > 1:
        raise SystemExit("--output names a single file; drop it or use -n 1.")
    seeds = (
        [args.seed] if args.seed is not None
        else [int(s) for s in np.random.randint(0, 2**31 - 1, n_videos)]
    )

    if not args.benchmark:
        out_dir = os.path.dirname(args.output) if args.output else "videos"
        if out_dir:
            os.makedirs(out_dir, exist_ok=True)

    for i, seed in enumerate(seeds):
        print(f"Video {i + 1}/{n_videos} — {args.width}x{args.height} "
              f"{args.fps}fps x {args.duration}s, seed {seed}")
        plan, chain, fps, seed = draw_plan_within_budget(seed, weights, args, probe_chunks)
        print(f"  tree: {' -> '.join(chain)}")
        print(f"  generation: {fps:.1f} fps on {args.processes} workers "
              f"({fps / args.fps:.2f}x real time, encoder not included)")

        if args.benchmark:
            continue

        output = args.output or f"videos/fast-{i + 1}.{args.extension}"
        cmd = build_ffmpeg_cmd(
            args.width, args.height, args.fps, codec, output,
            bitrate=args.bitrate,
            out_pix_fmt=_COMPAT_PIX_FMT if codec in _H264_CODECS else None,
            extra_args=["-preset", args.preset] if codec in _PRESET_CODECS else (),
        )
        t0 = time.perf_counter()
        try:
            run_ffmpeg_pipeline(cmd, args.processes, chunk_steps, compute_chunk,
                                pool_initializer=init_worker,
                                pool_initargs=(plan, args.color_space))
        except KeyboardInterrupt:
            break
        wall = time.perf_counter() - t0
        print(f"Done: {output} — {n_frames} frames in {wall:.1f}s "
              f"({n_frames / wall:.1f} fps, {args.duration / wall:.2f}x real time)")


if __name__ == "__main__":
    main()
