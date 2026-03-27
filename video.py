import os
import time
import numpy as np
import subprocess
import optparse
import multiprocessing as mp
from numpy.random import rand
from config import load_personality_list
from build import get_random_function
from functions_numba import UNBATCHED_1ARG, UNBATCHED_2ARG, NUMBA_SAFE

try:
    import numba
    NUMBA_AVAILABLE = True
except ImportError:
    NUMBA_AVAILABLE = False

p = load_personality_list("personality.json")
FFMPEG_BIN = os.getenv("FFMPEG_BIN")

if FFMPEG_BIN and not (FFMPEG_BIN + os.pathsep) in os.environ["PATH"]:
    os.environ["PATH"] = (FFMPEG_BIN + os.pathsep) + os.environ["PATH"]


def random_delta(tensor, alpha=5e-3):
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
                return [leaf, np.float32(random_delta(leaf, alpha))]
        except Exception as e:
            print(func.__name__, str(e))
            raise e
    return _build_tree(depth=0)



def _tree_to_source(tree, registry, counter):
    """Recursively generate a flat list of assignment statements for the tree.

    Returns (var_name, statements, is_numba_safe).
    """
    nc = counter[0]
    counter[0] += 1
    var = f'_t{nc}'

    if len(tree) == 2:
        base, delta = tree
        bname, dname = f'_base{nc}', f'_delta{nc}'
        registry[bname] = base
        registry[dname] = delta
        return var, [f'{var} = {bname} + {dname} * steps'], True

    _, func, branches = tree
    stmts, child_vars, all_safe = [], [], True

    for branch in branches:
        cvar, cstmts, csafe = _tree_to_source(branch, registry, counter)
        stmts.extend(cstmts)
        child_vars.append(cvar)
        all_safe = all_safe and csafe

    args_str = ', '.join(child_vars)

    if func in UNBATCHED_1ARG:
        fname = f'_func{nc}'
        registry[fname] = func
        c = child_vars[0]
        stmts.append(f'{var} = np.stack([{fname}({c}[i]) for i in range({c}.shape[0])])')
        return var, stmts, False

    elif func in UNBATCHED_2ARG:
        fname = f'_func{nc}'
        registry[fname] = func
        ca, cb = child_vars[0], child_vars[1]
        stmts.append(f'{var} = np.stack([{fname}({ca}[i], {cb}[i]) for i in range({ca}.shape[0])])')
        return var, stmts, False

    elif func in NUMBA_SAFE:
        stmts.extend(NUMBA_SAFE[func](var, child_vars))
        return var, stmts, all_safe

    else:
        fname = f'_func{nc}'
        registry[fname] = func
        stmts.append(f'{var} = {fname}({args_str})')
        return var, stmts, False


def compile_tree(tree, use_numba=True):
    """Compile tree to a callable, JIT-compiled with Numba if the tree is safe."""
    registry = {'np': np}
    result_var, stmts, is_numba_safe = _tree_to_source(tree, registry, [0])

    body = '\n    '.join(stmts)
    src = f'def _compiled(steps):\n    {body}\n    return {result_var}\n'

    print(f"Compiled source: {len(src.encode())} bytes. Starting to compile.")
    t0 = time.time()
    exec(src, registry)
    fn = registry['_compiled']

    if use_numba and is_numba_safe and NUMBA_AVAILABLE:
        try:
            fn = numba.jit(nopython=True, cache=False)(fn)
            print("Tree compiled with Numba (nopython mode)")
        except Exception as e:
            print(f"Numba failed ({e}), using Python")
    else:
        reason = (
            "disabled by user" if not use_numba
            else "exotic functions" if not is_numba_safe
            else "Numba not installed"
        )
        print(f"Using Python ({reason})")
    print(f"Finished compilation in {time.time() - t0:.2f}s")

    return fn


_worker_fn = None

def _init_worker(tree, use_numba):
    global _worker_fn
    _worker_fn = compile_tree(tree, use_numba=use_numba)

def _compute_chunk(steps):
    frames = _worker_fn(steps)
    return np.rint(frames.clip(0.0, 1.0) * 255.0).astype(np.uint8)


def parse_cmd_args():
    parser = optparse.OptionParser()
    parser.add_option('-n', '--n_videos', dest='n_videos', type=int, default=1, help='Number of videos. Default: 1.')
    parser.add_option('-f', '--fps', dest='fps', type=int, default=30, help='FPS. Default: 30.')
    parser.add_option('-H', '--height', dest='height', type=int, default=256, help='Video height. Default: 256.')
    parser.add_option('-W', '--width', dest='width', type=int, default=256, help='Video width. Default: 256.')
    parser.add_option('-s', '--step', dest='step', type=float, default=0.003, help='Alpha step for image generation. Default: 3e-3.')
    parser.add_option('-d', '--duration', dest='duration', type=int, default=10, help='Video duration. Default: 10.')
    parser.add_option('-S', '--seed', dest='seed', type=int, default=None, help='Seed. Default: None.')
    parser.add_option('-e', '--extension', dest='ext', type=str, default="webm", help='Extension. Can be webm, avi, mp4, gif, flv, ogg, mpeg. Default: webm.')
    parser.add_option('-b', '--bitrate', dest='bitrate', type=str, default="6M", help='Constant bitrate. Default: 6M.')
    parser.add_option('-C', '--codec', dest='codec', type=str, default=None, help='Video codec. Defaults to recommended codec for the chosen extension.')
    parser.add_option('-c', '--chunk_size', dest='chunk_size', type=int, default=10, help='Frames per batch. Lower values use less memory. Default: 10.')
    parser.add_option('-p', '--processes', dest='n_process', type=int, default=3, help='Number of parallel workers. Default: 3.')
    parser.add_option('--no-numba', dest='use_numba', action='store_false', default=True, help='Disable Numba JIT compilation.')
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

    recommended = RECOMMENDED_CODECS.get(args.ext, "libopenh264")
    if args.codec is None:
        args.codec = recommended
    elif args.codec != recommended:
        print(f"Warning: '{args.codec}' is not the recommended codec for .{args.ext}.")
        print(f"  Recommended: '{recommended}'. The video may not play properly.")

    os.makedirs("videos", exist_ok=True)

    n_videos = args.n_videos if args.seed is None else 1

    videofiles = os.listdir("videos")
    start_i = max([int(x.split("-")[1].split(".")[0]) for x in videofiles]) + 1 if videofiles else 1

    n_frames = args.fps * args.duration
    all_steps = (np.arange(n_frames) / args.fps).astype(np.float32).reshape(-1, 1, 1, 1)
    chunk_steps = [all_steps[s:s + args.chunk_size] for s in range(0, n_frames, args.chunk_size)]

    for i, video_n in enumerate(rand(n_videos)):
        video_n = int(video_n * 1e9) if args.seed is None else args.seed

        print(f"Building tree for video {i+1}/{n_videos} (seed={video_n})")
        tree = build_tree(min_depth=6, max_depth=16, seed=video_n, weights=p, dx=args.width, dy=args.height, alpha=4e-3)

        output_path = f"videos/video-{i+start_i}.{args.ext}"
        bitrate_args = ["-b:v", args.bitrate] if args.bitrate else []
        openh264_args = (
            ["-profile:v", "high", "-coder", "cabac", "-rc_mode", "bitrate"]
            if args.codec == "libopenh264" else []
        )
        ffmpeg_cmd = [
            "ffmpeg", "-y",
            "-f", "rawvideo",
            "-pixel_format", "rgb24",
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
        with mp.Pool(args.n_process, initializer=_init_worker, initargs=(tree, args.use_numba)) as pool:
            with subprocess.Popen(ffmpeg_cmd, stdin=subprocess.PIPE) as proc:
                for frames in pool.imap(_compute_chunk, chunk_steps):
                    proc.stdin.write(frames.tobytes())
        print(f"Done: {output_path}")
