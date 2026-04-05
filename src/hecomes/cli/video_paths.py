"""Path-based video generation CLI (``hecomes-video-paths``).

Parallel to ``hecomes-video`` but uses the path animation backend.  Leaves are
re-rendered each chunk from animated parameters rather than being pre-rendered
once and drifted via a delta.

Key differences from ``hecomes-video``:

* Trees are compiled with :func:`~hecomes.artgen.tree_paths.compile_plan_paths`.
* ODE paths are pre-integrated sequentially before the pool starts.
* ``_compute_chunk`` calls :func:`~hecomes.artgen.tree_paths.eval_plan_paths`.
* Extra CLI options: ``--path-personality``, ``--ode-solver``, ``--no-paths``.
"""

import multiprocessing as mp
import optparse
import os
import subprocess

import numpy as np
from numpy.random import rand

from hecomes.artgen.func_utils import hsv_to_rgb
from hecomes.artgen.tree import linearize
from hecomes.artgen.tree_paths import (
    build_node_paths,
    compile_plan_paths,
    eval_plan_paths,
    integrate_ode_paths,
    load_paths_config,
)
from hecomes.config import PERSONALITIES_DIR, load_personality_list

RECOMMENDED_CODECS = {
    "mp4": "libopenh264",
    "avi": "mpeg4",
    "webm": "libvpx-vp9",
    "mkv": "libopenh264",
    "ogg": "libtheora",
    "flv": "flv",
    "mpeg": "mpeg2video",
    "gif": "gif",
}

ALPHA_PIX_FMTS = {
    "libvpx-vp9": "yuva420p",
    "gif": "pal8",
}

# ── Process-level globals (shared across pool workers via fork) ───────────────
# Same structure as video.py globals; _plans replaces _trees.

_plans = None           # list of compiled path plans (one per color/extra tree)
_color_space = "rgb"
_independent_channels = False
_n_color = 1
_use_gpu = False


# ── Internal helpers ──────────────────────────────────────────────────────────


def _build(min_depth, max_depth, dx, dy, weights, seed, paths_config, duration, alpha):
    np.random.seed(seed % (2**32 - 1))
    nodes, paths_per_leaf, params_per_leaf = {}, {}, {}
    root_id = build_node_paths(
        0, min_depth, max_depth, dx, dy, weights,
        nodes, paths_per_leaf, params_per_leaf, paths_config, duration,
    )
    order = linearize(root_id, nodes)
    return compile_plan_paths(order, nodes, paths_per_leaf, params_per_leaf, dx, dy)


def _compute_chunk(steps):
    if _independent_channels:
        channels = [
            eval_plan_paths(_plans[i], steps, use_gpu=_use_gpu)[..., i : i + 1]
            for i in range(3)
        ]
        raw = np.concatenate(channels, axis=-1)
    else:
        raw = eval_plan_paths(_plans[0], steps, use_gpu=_use_gpu)

    extra = [
        eval_plan_paths(_plans[i], steps, use_gpu=_use_gpu)[..., 0:1].clip(0, 1)
        for i in range(_n_color, len(_plans))
    ]

    if _color_space == "hsv":
        hsv = np.stack(
            [raw[..., 0] % 1.0, raw[..., 1].clip(0, 1), raw[..., 2].clip(0, 1)],
            axis=-1,
        )
        frames = hsv_to_rgb(hsv)
    elif _color_space == "cmy":
        k = extra.pop(0) if extra else 0.0
        frames = (1.0 - raw.clip(0, 1)) * (1.0 - k)
    else:
        frames = raw.clip(0, 1)

    if extra:
        frames = np.concatenate([frames, extra[0]], axis=-1)

    return np.rint(frames * 255.0).astype(np.uint8)


# ── Argument parsing ──────────────────────────────────────────────────────────


def _parse_args():
    parser = optparse.OptionParser()
    parser.add_option("-n", "--n_videos", dest="n_videos", type=int, default=1,
                      help="Number of videos. Default: 1.")
    parser.add_option("-f", "--fps", dest="fps", type=int, default=30,
                      help="FPS. Default: 30.")
    parser.add_option("-s", "--step", dest="step", type=float, default=1e-1,
                      help="Step size multiplier. Default: 1e-2.")
    parser.add_option("-H", "--height", dest="height", type=int, default=256,
                      help="Video height. Default: 256.")
    parser.add_option("-W", "--width", dest="width", type=int, default=256,
                      help="Video width. Default: 256.")
    parser.add_option("-d", "--duration", dest="duration", type=int, default=10,
                      help="Video duration in seconds. Default: 10.")
    parser.add_option("-S", "--seed", dest="seed", type=int, default=None,
                      help="Seed. Default: None (random).")
    parser.add_option("-e", "--extension", dest="ext", type=str, default="webm",
                      help="Extension: webm, avi, mp4, gif, flv, ogg, mpeg. Default: webm.")
    parser.add_option("-b", "--bitrate", dest="bitrate", type=str, default="6M",
                      help="Constant bitrate. Default: 6M.")
    parser.add_option("-C", "--codec", dest="codec", type=str, default=None,
                      help="Video codec. Defaults to recommended for the chosen extension.")
    parser.add_option("-c", "--chunk_size", dest="chunk_size", type=int, default=10,
                      help="Frames per batch. Default: 10.")
    parser.add_option("-p", "--processes", dest="n_process", type=int, default=3,
                      help="Number of parallel workers. Default: 3.")
    parser.add_option("--min-depth", dest="min_depth", type=int, default=6,
                      help="Minimum tree depth. Default: 6.")
    parser.add_option("--max-depth", dest="max_depth", type=int, default=12,
                      help="Maximum tree depth. Default: 12.")
    parser.add_option("--color-space", dest="color_space", type=str, default="rgb",
                      help="Color space: rgb, hsv, cmy. Default: rgb.")
    parser.add_option("--independent-channels", dest="independent_channels",
                      action="store_true", default=False,
                      help="Build one tree per channel.")
    parser.add_option("--k", dest="k", action="store_true", default=False,
                      help="Generate K channel from a tree (CMY mode only).")
    parser.add_option("--alpha", dest="alpha", action="store_true", default=False,
                      help="Generate alpha channel from a tree.")
    parser.add_option("--personality", dest="personality", type=str, default=None,
                      help="Personality name in data/personalities/. Default: personality.")
    parser.add_option("--path-personality", dest="path_personality", type=str, default=None,
                      help=(
                          "Path to a personality JSON with a 'paths' section. "
                          "Defaults to the same file as --personality. "
                          "Use a separate file to mix tree and path personalities."
                      ))
    parser.add_option("--gpu", dest="gpu", action="store_true", default=False,
                      help="Evaluate inner-node ops on GPU via CuPy. Default: CPU.")
    parser.add_option("--ode-solver", dest="ode_solver", type=str, default="rk4",
                      help="ODE integration method: euler or rk4. Default: rk4.")
    parser.add_option("--no-paths", dest="no_paths", action="store_true", default=False,
                      help=(
                          "Build path trees but disable all path animation "
                          "(leaves use build-time params only). Useful for debugging."
                      ))
    args, _ = parser.parse_args()
    return args


# ── Entry point ───────────────────────────────────────────────────────────────


def main():
    global _plans, _color_space, _independent_channels, _n_color, _use_gpu

    ffmpeg_bin = os.getenv("FFMPEG_BIN")
    if ffmpeg_bin and (ffmpeg_bin + os.pathsep) not in os.environ["PATH"]:
        os.environ["PATH"] = (ffmpeg_bin + os.pathsep) + os.environ["PATH"]

    args = _parse_args()

    _use_gpu = args.gpu
    if _use_gpu and args.n_process > 1:
        print("Warning: --gpu with multiple workers splits VRAM across processes. Consider --processes 1.")

    if args.color_space not in ("rgb", "hsv", "cmy"):
        raise ValueError(f"Unknown color space '{args.color_space}'. Choose from: rgb, hsv, cmy.")
    if args.k and args.color_space != "cmy":
        print("Warning: --k is only meaningful in cmy mode, ignoring.")
        args.k = False

    personality = args.personality or "personality"
    p = load_personality_list(PERSONALITIES_DIR / (personality + ".json"))
    p_h = load_personality_list(PERSONALITIES_DIR / "hsv.json")

    # Load paths config from --path-personality (falls back to --personality file)
    path_personality_file = (
        args.path_personality
        or str(PERSONALITIES_DIR / (personality + ".json"))
    )
    paths_config = load_paths_config(path_personality_file)
    if args.no_paths:
        paths_config = dict(paths_config, animation_probability=0.0)

    recommended = RECOMMENDED_CODECS.get(args.ext, "libopenh264")
    if args.codec is None:
        args.codec = recommended
    elif args.codec != recommended:
        print(f"Warning: '{args.codec}' is not the recommended codec for .{args.ext}.")
        print(f"  Recommended: '{recommended}'. The video may not play properly.")

    os.makedirs("videos", exist_ok=True)

    n_videos = args.n_videos if args.seed is None else 1

    videofiles = os.listdir("videos")
    start_i = (
        max([int(x.split("-")[1].split(".")[0]) for x in videofiles]) + 1
        if videofiles
        else 1
    )

    n_frames = args.fps * args.duration
    duration = float(args.duration)
    all_steps = (np.arange(n_frames) * args.step / args.fps).astype(np.float32).reshape(-1, 1, 1, 1)
    chunk_steps = [
        all_steps[s : s + args.chunk_size] for s in range(0, n_frames, args.chunk_size)
    ]

    pixel_format = "rgba" if args.alpha else "rgb24"

    build_kwargs = dict(
        min_depth=args.min_depth,
        max_depth=args.max_depth,
        dx=args.width,
        dy=args.height,
        paths_config=paths_config,
        duration=duration,
        alpha=4e-3,
    )

    for i, video_n in enumerate(rand(n_videos)):
        video_n = int(video_n * 1e9) if args.seed is None else args.seed

        if args.color_space == "hsv" and args.independent_channels:
            print(f"Building H/S/V path trees for video {i+1}/{n_videos} (seed={video_n})")
            color_plans = [
                _build(weights=p_h, seed=video_n,          **build_kwargs),
                _build(weights=p,   seed=video_n ^ 0xABCD, **build_kwargs),
                _build(weights=p,   seed=video_n ^ 0x1234, **build_kwargs),
            ]
            _n_color = 3
        else:
            print(f"Building path tree for video {i+1}/{n_videos} (seed={video_n})")
            color_plans = [_build(weights=p, seed=video_n, **build_kwargs)]
            _n_color = 1

        extra_plans = []
        if args.k:
            extra_plans.append(_build(weights=p, seed=video_n ^ 0x5678, **build_kwargs))
        if args.alpha:
            extra_plans.append(_build(weights=p, seed=video_n ^ 0x9ABC, **build_kwargs))

        _plans = color_plans + extra_plans
        _color_space = args.color_space
        _independent_channels = args.independent_channels

        # Pre-integrate any ODE paths sequentially before distributing chunks
        print("Integrating ODE paths...")
        for plan in _plans:
            integrate_ode_paths(plan, n_frames, args.fps, solver=args.ode_solver)

        output_path = f"videos/video-{i+start_i}.{args.ext}"
        bitrate_args = ["-b:v", args.bitrate] if args.bitrate else []
        openh264_args = (
            ["-profile:v", "high", "-coder", "cabac", "-rc_mode", "bitrate"]
            if args.codec == "libopenh264"
            else []
        )
        if args.alpha:
            if args.codec in ALPHA_PIX_FMTS:
                alpha_args = ["-pix_fmt", ALPHA_PIX_FMTS[args.codec]]
            else:
                print(f"Warning: codec '{args.codec}' does not support alpha. Alpha channel will be dropped.")
                alpha_args = []
        else:
            alpha_args = []

        ffmpeg_cmd = [
            "ffmpeg", "-y",
            "-f", "rawvideo",
            "-pixel_format", pixel_format,
            "-video_size", f"{args.width}x{args.height}",
            "-framerate", str(args.fps),
            "-i", "pipe:0",
            "-vcodec", args.codec,
            *openh264_args,
            *alpha_args,
            *bitrate_args,
            output_path,
        ]

        print("Running:\n\t{}".format(" ".join(ffmpeg_cmd)))
        print(f"Encoding {n_frames} frames to {output_path}")
        try:
            with mp.Pool(args.n_process) as pool:
                with subprocess.Popen(ffmpeg_cmd, stdin=subprocess.PIPE) as proc:
                    try:
                        for frames in pool.imap(_compute_chunk, chunk_steps):
                            proc.stdin.write(frames.tobytes())
                    except KeyboardInterrupt:
                        print("\nInterrupted — closing FFmpeg pipe...")
                        pool.terminate()
                        proc.stdin.close()
                        proc.wait()
                        raise
        except KeyboardInterrupt:
            break
        print(f"Done: {output_path}")


if __name__ == "__main__":
    main()
