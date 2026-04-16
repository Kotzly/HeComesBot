"""
Shared utilities for the video-generating CLI entrypoints (video, video_paths, instagram).

Centralises code that was previously duplicated across all three:
  - RECOMMENDED_CODECS / ALPHA_PIX_FMTS constants
  - FFmpeg PATH setup, codec selection, command building, pipeline execution
  - Path-backend plan builder, worker initialiser, and chunk compute function
    (video_paths.py and instagram.py share the identical path-eval loop)
"""

import multiprocessing as mp
import os
import subprocess

import numpy as np

from hecomes.artgen.func_utils import hsv_to_rgb
from hecomes.artgen.tree import linearize
from hecomes.artgen.tree_paths import (
    build_node_paths,
    compile_plan_paths,
    eval_plan_paths,
)

# ── Codec tables ──────────────────────────────────────────────────────────────

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

# Output pixel formats that preserve alpha, keyed by codec.
ALPHA_PIX_FMTS = {
    "libvpx-vp9": "yuva420p",
    "gif": "pal8",
}


# ── FFmpeg helpers ────────────────────────────────────────────────────────────

def setup_ffmpeg_path():
    """Prepend FFMPEG_BIN env var to PATH if set and not already present."""
    ffmpeg_bin = os.getenv("FFMPEG_BIN")
    if ffmpeg_bin and (ffmpeg_bin + os.pathsep) not in os.environ.get("PATH", ""):
        os.environ["PATH"] = ffmpeg_bin + os.pathsep + os.environ["PATH"]


def select_codec(ext, requested_codec):
    """Return the codec to use, printing a warning if it differs from the recommendation."""
    recommended = RECOMMENDED_CODECS.get(ext, "libopenh264")
    if requested_codec is None:
        return recommended
    if requested_codec != recommended:
        print(f"Warning: '{requested_codec}' is not the recommended codec for .{ext}.")
        print(f"  Recommended: '{recommended}'. The video may not play properly.")
    return requested_codec


def build_ffmpeg_cmd(width, height, fps, codec, output_path,
                     pixel_format="rgb24", bitrate="6M", alpha_pix_fmt=None):
    """Build an ffmpeg command list that reads raw frames from stdin."""
    openh264_args = (
        ["-profile:v", "high", "-coder", "cabac", "-rc_mode", "bitrate"]
        if codec == "libopenh264" else []
    )
    bitrate_args = ["-b:v", bitrate] if bitrate else []
    alpha_args = ["-pix_fmt", alpha_pix_fmt] if alpha_pix_fmt else []
    return [
        "ffmpeg", "-y",
        "-f", "rawvideo",
        "-pixel_format", pixel_format,
        "-video_size", f"{width}x{height}",
        "-framerate", str(fps),
        "-i", "pipe:0",
        "-vcodec", codec,
        *openh264_args,
        *alpha_args,
        *bitrate_args,
        output_path,
    ]


def run_ffmpeg_pipeline(ffmpeg_cmd, n_process, chunk_steps, compute_fn,
                        pool_initializer=None, pool_initargs=()):
    """
    Stream rendered frames from a worker pool into an ffmpeg subprocess.

    Raises KeyboardInterrupt (after cleanup) if interrupted mid-encoding so
    callers in a multi-video loop can decide whether to continue (``pass``) or
    stop (``break``).
    """
    print("Running:\n\t" + " ".join(ffmpeg_cmd))
    pool_kwargs = (
        {"initializer": pool_initializer, "initargs": pool_initargs}
        if pool_initializer is not None else {}
    )
    with mp.Pool(n_process, **pool_kwargs) as pool:
        with subprocess.Popen(ffmpeg_cmd, stdin=subprocess.PIPE) as proc:
            try:
                for frames in pool.imap(compute_fn, chunk_steps):
                    proc.stdin.write(frames.tobytes())
            except KeyboardInterrupt:
                print("\nInterrupted — closing FFmpeg pipe...")
                pool.terminate()
                proc.stdin.close()
                proc.wait()
                raise


# ── Path-backend: plan builder, worker globals, chunk compute ─────────────────
#
# Shared between video_paths.py and instagram.py, which use an identical
# path-animation eval loop.  Lives here so pool workers can pickle the compute
# function (must be a top-level name in an importable module) and so the
# pool-initialiser pattern replaces per-module global assignment in each caller.

_plans = None
_color_space = "rgb"
_independent_channels = False
_n_color = 1
_use_gpu = False


def init_worker(plans, color_space, independent_channels, n_color, use_gpu):
    """Pool initialiser — sets per-worker module globals at fork/spawn time."""
    global _plans, _color_space, _independent_channels, _n_color, _use_gpu
    _plans = plans
    _color_space = color_space
    _independent_channels = independent_channels
    _n_color = n_color
    _use_gpu = use_gpu


def compute_chunk_paths(steps):
    """Evaluate one chunk of frames using the path-animation backend."""
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


def build_path_plan(min_depth, max_depth, dx, dy, weights, seed,
                    paths_config, duration, alpha):
    """Build and compile one path-animation plan."""
    np.random.seed(seed % (2**32 - 1))
    nodes, paths_per_leaf, params_per_leaf = {}, {}, {}
    root_id = build_node_paths(
        0, min_depth, max_depth, dx, dy, weights,
        nodes, paths_per_leaf, params_per_leaf, paths_config, duration,
    )
    order = linearize(root_id, nodes)
    return compile_plan_paths(order, nodes, paths_per_leaf, params_per_leaf, dx, dy)
