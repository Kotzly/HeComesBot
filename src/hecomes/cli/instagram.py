"""Instagram posting CLI (``hecomes-instagram``).

When ``--url`` is given, posts that URL directly — no generation needed.
Otherwise generates art using the path-animation backend, then posts it.

Credentials are read from ``~/.hecomes_instagram.json`` or environment
variables.  See ``hecomes.instagram.poster.load_credentials`` for details.

Example usage::

    # Generate and post a square image
    hecomes-instagram --type image --caption "Generated art"

    # Generate and post a Reel (9:16, 15 s)
    hecomes-instagram --type reel -W 540 -H 960 -d 15 --caption "Animation"

    # Post an existing public URL (no generation)
    hecomes-instagram --url https://example.com/photo.jpg --caption "My art"
    hecomes-instagram --url https://example.com/clip.mp4 --type reel
    hecomes-instagram --url https://example.com/story.jpg --type story-image
"""

import multiprocessing as mp
import optparse
import os
import subprocess
import tempfile

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
from hecomes.instagram.poster import InstagramPoster, load_credentials

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

VALID_TYPES = ("image", "reel", "story-image", "story-video")

# ── Process-level globals (shared across pool workers via fork) ───────────────

_plans = None
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


def _generate_video(args, plans, output_path):
    global _plans, _color_space, _independent_channels, _n_color, _use_gpu

    n_frames = args.fps * args.duration
    all_steps = (
        np.arange(n_frames) * args.step / args.fps
    ).astype(np.float32).reshape(-1, 1, 1, 1)
    chunk_steps = [
        all_steps[s : s + args.chunk_size]
        for s in range(0, n_frames, args.chunk_size)
    ]

    _plans = plans
    _color_space = args.color_space
    _independent_channels = args.independent_channels
    _use_gpu = args.gpu

    print("Integrating ODE paths...")
    for plan in _plans:
        integrate_ode_paths(plan, n_frames, args.fps, solver=args.ode_solver)

    recommended = RECOMMENDED_CODECS.get(args.ext, "libopenh264")
    codec = args.codec or recommended
    openh264_args = (
        ["-profile:v", "high", "-coder", "cabac", "-rc_mode", "bitrate"]
        if codec == "libopenh264"
        else []
    )
    bitrate_args = ["-b:v", args.bitrate] if args.bitrate else []

    ffmpeg_cmd = [
        "ffmpeg", "-y",
        "-f", "rawvideo",
        "-pixel_format", "rgb24",
        "-video_size", f"{args.width}x{args.height}",
        "-framerate", str(args.fps),
        "-i", "pipe:0",
        "-vcodec", codec,
        *openh264_args,
        *bitrate_args,
        output_path,
    ]

    print("Running:\n\t{}".format(" ".join(ffmpeg_cmd)))
    print(f"Encoding {n_frames} frames to {output_path}")
    with mp.Pool(args.n_process) as pool:
        with subprocess.Popen(ffmpeg_cmd, stdin=subprocess.PIPE) as proc:
            try:
                for frames in pool.imap(_compute_chunk, chunk_steps):
                    proc.stdin.write(frames.tobytes())
            except KeyboardInterrupt:
                pool.terminate()
                proc.stdin.close()
                proc.wait()
                raise


def _generate_image(args, seed, output_path):
    from PIL import Image

    personality = args.personality or "personality"
    p = load_personality_list(PERSONALITIES_DIR / (personality + ".json"))
    path_personality_file = (
        args.path_personality
        or str(PERSONALITIES_DIR / (personality + ".json"))
    )
    paths_config = load_paths_config(path_personality_file)

    np.random.seed(seed % (2**32 - 1))
    nodes, paths_per_leaf, params_per_leaf = {}, {}, {}
    root_id = build_node_paths(
        0, args.min_depth, args.max_depth, args.width, args.height,
        p, nodes, paths_per_leaf, params_per_leaf, paths_config,
        duration=1.0,
    )
    order = linearize(root_id, nodes)
    plan = compile_plan_paths(
        order, nodes, paths_per_leaf, params_per_leaf, args.width, args.height
    )
    t = np.zeros((1, 1, 1, 1), dtype=np.float32)
    frame = eval_plan_paths(plan, t)[0]
    img_arr = np.rint(frame.clip(0, 1) * 255).astype(np.uint8)
    Image.fromarray(img_arr).save(output_path)
    print(f"Saved image to {output_path}")


# ── Post from URL (no generation) ────────────────────────────────────────────


def _post_url(poster, post_type, url, caption):
    if post_type == "image":
        cid = poster._create_container(image_url=url, caption=caption)
    elif post_type == "reel":
        cid = poster._create_container(video_url=url, caption=caption, media_type="REELS")
    elif post_type == "story-image":
        cid = poster._create_container(image_url=url, media_type="STORIES")
    else:  # story-video
        cid = poster._create_container(video_url=url, media_type="STORIES")
    poster._wait(cid)
    result = poster._publish(cid)
    print(f"Posted — media ID: {result.get('id')}")


# ── Argument parsing ──────────────────────────────────────────────────────────


def _parse_args():
    parser = optparse.OptionParser(
        usage="hecomes-instagram [options]",
        description=(
            "Generate art and post it to Instagram, or post an existing public URL. "
            "Requires credentials in ~/.hecomes_instagram.json or env vars."
        ),
    )
    parser.add_option("--url", dest="url", type=str, default=None,
                      help="Post an existing public URL instead of generating art.")
    parser.add_option("--type", dest="post_type", type=str, default="image",
                      help=f"Post type: {', '.join(VALID_TYPES)}. Default: image.")
    parser.add_option("--caption", dest="caption", type=str, default="",
                      help="Post caption (not used for stories). Default: empty.")
    parser.add_option("--credentials", dest="credentials", type=str, default=None,
                      help="Path to credentials JSON. Default: ~/.hecomes_instagram.json.")
    parser.add_option("-S", "--seed", dest="seed", type=int, default=None,
                      help="Random seed. Default: random.")
    parser.add_option("-W", "--width", dest="width", type=int, default=512,
                      help="Width in pixels. Default: 512.")
    parser.add_option("-H", "--height", dest="height", type=int, default=512,
                      help="Height in pixels. Default: 512.")
    parser.add_option("--min-depth", dest="min_depth", type=int, default=6)
    parser.add_option("--max-depth", dest="max_depth", type=int, default=12)
    parser.add_option("--personality", dest="personality", type=str, default=None)
    parser.add_option("--path-personality", dest="path_personality", type=str, default=None)
    parser.add_option("-f", "--fps", dest="fps", type=int, default=30)
    parser.add_option("-d", "--duration", dest="duration", type=int, default=15,
                      help="[reel/story-video] Duration in seconds. Default: 15.")
    parser.add_option("-s", "--step", dest="step", type=float, default=0.1)
    parser.add_option("-e", "--extension", dest="ext", type=str, default="mp4")
    parser.add_option("-b", "--bitrate", dest="bitrate", type=str, default="6M")
    parser.add_option("-C", "--codec", dest="codec", type=str, default=None)
    parser.add_option("-c", "--chunk_size", dest="chunk_size", type=int, default=10)
    parser.add_option("-p", "--processes", dest="n_process", type=int, default=3)
    parser.add_option("--color-space", dest="color_space", type=str, default="rgb")
    parser.add_option("--independent-channels", dest="independent_channels",
                      action="store_true", default=False)
    parser.add_option("--ode-solver", dest="ode_solver", type=str, default="rk4")
    parser.add_option("--gpu", dest="gpu", action="store_true", default=False)
    args, _ = parser.parse_args()
    return args


# ── Entry point ───────────────────────────────────────────────────────────────


def main():
    args = _parse_args()

    if args.post_type not in VALID_TYPES:
        raise ValueError(f"--type must be one of {VALID_TYPES}, got '{args.post_type}'")

    ffmpeg_bin = os.getenv("FFMPEG_BIN")
    if ffmpeg_bin and (ffmpeg_bin + os.pathsep) not in os.environ.get("PATH", ""):
        os.environ["PATH"] = (ffmpeg_bin + os.pathsep) + os.environ["PATH"]

    creds = load_credentials(args.credentials)
    poster = InstagramPoster(**creds)

    # ── URL mode: skip generation entirely ───────────────────────────────────
    if args.url:
        _post_url(poster, args.post_type, args.url, args.caption)
        return

    # ── Generation mode ───────────────────────────────────────────────────────
    seed = args.seed if args.seed is not None else int(rand() * 1e9)
    print(f"Seed: {seed}")

    if args.post_type == "image":
        with tempfile.NamedTemporaryFile(suffix=".png", delete=False) as f:
            tmp_path = f.name
        try:
            _generate_image(args, seed, tmp_path)
            poster.post_image(tmp_path, caption=args.caption)
        finally:
            os.unlink(tmp_path)

    elif args.post_type in ("reel", "story-video"):
        is_reel = args.post_type == "reel"
        if is_reel:
            if args.duration < 5 or args.duration > 90:
                print(f"Warning: Reel duration {args.duration}s is outside 5–90s.")
            if args.width / args.height != 9 / 16:
                print("Warning: Reels recommend 9:16 aspect ratio (e.g. -W 540 -H 960).")
        else:
            if args.duration > 60:
                print(f"Warning: Story video duration {args.duration}s exceeds 60s limit.")
            if args.width / args.height != 9 / 16:
                print("Warning: Stories recommend 9:16 aspect ratio (e.g. -W 540 -H 960).")

        personality = args.personality or "personality"
        p = load_personality_list(PERSONALITIES_DIR / (personality + ".json"))
        p_h = load_personality_list(PERSONALITIES_DIR / "hsv.json")
        path_personality_file = (
            args.path_personality
            or str(PERSONALITIES_DIR / (personality + ".json"))
        )
        paths_config = load_paths_config(path_personality_file)

        build_kwargs = dict(
            min_depth=args.min_depth,
            max_depth=args.max_depth,
            dx=args.width,
            dy=args.height,
            paths_config=paths_config,
            duration=float(args.duration),
            alpha=4e-3,
        )
        print(f"Building path tree (seed={seed})")
        if args.color_space == "hsv" and args.independent_channels:
            color_plans = [
                _build(weights=p_h, seed=seed,          **build_kwargs),
                _build(weights=p,   seed=seed ^ 0xABCD, **build_kwargs),
                _build(weights=p,   seed=seed ^ 0x1234, **build_kwargs),
            ]
        else:
            color_plans = [_build(weights=p, seed=seed, **build_kwargs)]

        with tempfile.NamedTemporaryFile(suffix=f".{args.ext}", delete=False) as f:
            tmp_path = f.name
        try:
            _generate_video(args, color_plans, tmp_path)
            if is_reel:
                poster.post_reel(tmp_path, caption=args.caption)
            else:
                poster.post_story_video(tmp_path)
        finally:
            os.unlink(tmp_path)

    else:  # story-image
        if args.width / args.height != 9 / 16:
            print("Warning: Stories recommend 9:16 aspect ratio (e.g. -W 540 -H 960).")
        with tempfile.NamedTemporaryFile(suffix=".png", delete=False) as f:
            tmp_path = f.name
        try:
            _generate_image(args, seed, tmp_path)
            poster.post_story_image(tmp_path)
        finally:
            os.unlink(tmp_path)


if __name__ == "__main__":
    main()
