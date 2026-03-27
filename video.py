import os
import numpy as np
import subprocess
import optparse
import multiprocessing as mp
from numpy.random import rand
from config import load_personality_list
from build import get_random_function
from functions import hsv_to_rgb

FFMPEG_BIN = os.getenv("FFMPEG_BIN")

if FFMPEG_BIN and not (FFMPEG_BIN + os.pathsep) in os.environ["PATH"]:
    os.environ["PATH"] = (FFMPEG_BIN + os.pathsep) + os.environ["PATH"]


def random_delta(alpha=5e-3):
    return np.random.choice([1, -1]) * alpha


def build_tree(min_depth=5, max_depth=15, dx=100, dy=100, weights=None, seed=42, alpha=1e-2):
    np.random.seed(seed % (2**32 - 1))

    def _build_tree(depth=0):
        n_args, func = get_random_function(depth, p=weights, min_depth=min_depth, max_depth=max_depth)
        args = [_build_tree(depth + 1) for _ in range(n_args)]
        kwargs = dict(dx=dx, dy=dy) if n_args == 0 else {}
        try:
            if n_args != 0:
                return [n_args, func, args]
            else:
                leaf = func(*args, **kwargs).astype(np.float32)
                return [leaf, np.float32(random_delta(alpha))]
        except Exception as e:
            print(func.__name__, str(e))
            raise e
    return _build_tree(depth=0)


def eval_tree(tree, steps):
    if len(tree) == 2:
        base, delta = tree
        return base + delta * steps
    _, func, branches = tree
    args = [eval_tree(b, steps) for b in branches]
    return func(*args)


_trees = None
_color_space = 'rgb'
_independent_channels = False
_alpha = None
_k = 0.0


def _compute_chunk(steps):
    if _independent_channels:
        channels = [eval_tree(t, steps)[..., i:i+1] for i, t in enumerate(_trees)]
        raw = np.concatenate(channels, axis=-1)
    else:
        raw = eval_tree(_trees[0], steps)

    if _color_space == 'hsv':
        hsv = np.concatenate([
            raw[..., 0:1] % 1.0,
            raw[..., 1:2].clip(0, 1),
            raw[..., 2:3].clip(0, 1),
        ], axis=-1)
        frames = hsv_to_rgb(hsv)
    elif _color_space == 'cmy':
        frames = (1.0 - raw.clip(0, 1)) * (1.0 - _k)
    else:
        frames = raw.clip(0, 1)

    if _alpha is not None:
        alpha_ch = np.full(frames.shape[:-1] + (1,), _alpha, dtype=np.float32)
        frames = np.concatenate([frames, alpha_ch], axis=-1)

    return np.rint(frames * 255.0).astype(np.uint8)


def parse_cmd_args():
    parser = optparse.OptionParser()
    parser.add_option('-n', '--n_videos', dest='n_videos', type=int, default=1, help='Number of videos. Default: 1.')
    parser.add_option('-f', '--fps', dest='fps', type=int, default=30, help='FPS. Default: 30.')
    parser.add_option('-H', '--height', dest='height', type=int, default=256, help='Video height. Default: 256.')
    parser.add_option('-W', '--width', dest='width', type=int, default=256, help='Video width. Default: 256.')
    parser.add_option('-s', '--step', dest='step', type=float, default=0.003, help='Alpha step for image generation. Default: 3e-3.')
    parser.add_option('-d', '--duration', dest='duration', type=int, default=10, help='Video duration. Default: 10.')
    parser.add_option('-S', '--seed', dest='seed', type=int, default=None, help='Seed. Default: None.')
    parser.add_option('-e', '--extension', dest='ext', type=str, default="webm", help='Extension. Default: webm.')
    parser.add_option('-b', '--bitrate', dest='bitrate', type=str, default="6M", help='Constant bitrate. Default: 6M.')
    parser.add_option('-C', '--codec', dest='codec', type=str, default=None, help='Video codec. Defaults to recommended codec for the chosen extension.')
    parser.add_option('-c', '--chunk_size', dest='chunk_size', type=int, default=10, help='Frames per batch. Default: 10.')
    parser.add_option('-p', '--processes', dest='n_process', type=int, default=3, help='Number of parallel workers. Default: 3.')
    parser.add_option('--color-space', dest='color_space', type=str, default='rgb',
                      help='Color space: rgb, hsv, cmy. Default: rgb.')
    parser.add_option('--independent-channels', dest='independent_channels', action='store_true', default=False,
                      help='Build one tree per channel (HSV only: H uses personality_h.json, S/V use personality.json).')
    parser.add_option('--alpha', dest='alpha', type=float, default=None,
                      help='Fixed alpha (opacity) channel value in [0, 1]. Adds a 4th channel to the output.')
    parser.add_option('--k', dest='k', type=float, default=0.0,
                      help='Fixed K (black) channel value for CMY mode, in [0, 1]. Default: 0.')
    args, _ = parser.parse_args()
    return args


RECOMMENDED_CODECS = {
    "mp4":  "libopenh264",
    "avi":  "mpeg4",
    "webm": "libvpx-vp9",
    "mkv":  "libopenh264",
    "ogg":  "libtheora",
    "flv":  "flv",
    "mpeg": "mpeg2video",
    "gif":  "gif",
}

if __name__ == "__main__":
    args = parse_cmd_args()

    if args.color_space not in ('rgb', 'hsv', 'cmy'):
        raise ValueError(f"Unknown color space '{args.color_space}'. Choose from: rgb, hsv, cmy.")

    recommended = RECOMMENDED_CODECS.get(args.ext, "libopenh264")
    if args.codec is None:
        args.codec = recommended
    elif args.codec != recommended:
        print(f"Warning: '{args.codec}' is not the recommended codec for .{args.ext}.")
        print(f"  Recommended: '{recommended}'. The video may not play properly.")

    os.makedirs("videos", exist_ok=True)

    p = load_personality_list("personality.json")
    p_h = load_personality_list("personality_h.json")

    n_videos = args.n_videos if args.seed is None else 1

    videofiles = os.listdir("videos")
    start_i = max([int(x.split("-")[1].split(".")[0]) for x in videofiles]) + 1 if videofiles else 1

    n_frames = args.fps * args.duration
    all_steps = (np.arange(n_frames) / args.fps).astype(np.float32).reshape(-1, 1, 1, 1)
    chunk_steps = [all_steps[s:s + args.chunk_size] for s in range(0, n_frames, args.chunk_size)]

    pixel_format = "rgba" if args.alpha is not None else "rgb24"

    for i, video_n in enumerate(rand(n_videos)):
        video_n = int(video_n * 1e9) if args.seed is None else args.seed

        build_kwargs = dict(min_depth=6, max_depth=16, dx=args.width, dy=args.height, alpha=4e-3)

        if args.color_space == 'hsv' and args.independent_channels:
            print(f"Building H/S/V trees for video {i+1}/{n_videos} (seed={video_n})")
            _trees = [
                build_tree(seed=video_n,            weights=p_h, **build_kwargs),
                build_tree(seed=video_n ^ 0xABCD,   weights=p,   **build_kwargs),
                build_tree(seed=video_n ^ 0x1234,   weights=p,   **build_kwargs),
            ]
        else:
            print(f"Building tree for video {i+1}/{n_videos} (seed={video_n})")
            _trees = [build_tree(seed=video_n, weights=p, **build_kwargs)]

        _color_space = args.color_space
        _independent_channels = args.independent_channels
        _alpha = args.alpha
        _k = args.k

        output_path = f"videos/video-{i+start_i}.{args.ext}"
        bitrate_args = ["-b:v", args.bitrate] if args.bitrate else []
        openh264_args = (
            ["-profile:v", "high", "-coder", "cabac", "-rc_mode", "bitrate"]
            if args.codec == "libopenh264" else []
        )
        ffmpeg_cmd = [
            "ffmpeg", "-y",
            "-f", "rawvideo",
            "-pixel_format", pixel_format,
            "-video_size", f"{args.width}x{args.height}",
            "-framerate", str(args.fps),
            "-i", "pipe:0",
            "-vcodec", args.codec,
            *openh264_args,
            *bitrate_args,
            output_path
        ]

        print("Running:\n\t{}".format(' '.join(ffmpeg_cmd)))
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
        except KeyboardInterrupt:
            pass
        print(f"Done: {output_path}")
