import os
from build import *
import numpy as np
import sys
import subprocess
from config import load_personality_list
import optparse
from PIL import Image
import shutil
import time
from numpy.random import rand
import multiprocessing as mp

p = load_personality_list("personality.json")
FFMPEG_BIN = os.getenv("FFMPEG_BIN")

if not (FFMPEG_BIN + ";") in os.environ["PATH"]:
    os.environ["PATH"] = (FFMPEG_BIN + ";") + os.environ["PATH"]

def random_delta(tensor, alpha=5e-3):
    random_direction = np.random.choice([1, -1]) * alpha
    return random_direction

def build_tree(min_depth=5, max_depth=15, dx=100, dy=100, weights=None, log_filepath="tree.txt", seed=42, alpha=1e-2):
    
    np.random.seed(seed % (2**32 - 1))
    
    def _build_tree(depth=0):
        n_args, func = get_random_function(depth, p=weights, min_depth=min_depth, max_depth=max_depth)
        args = [_build_tree(depth + 1) for i in range(n_args)]
        kwargs = dict(dx=dx, dy=dy) if n_args == 0 else {}
        try:
            if n_args != 0:
                return [n_args, func, args]
            else:
                leaf = func(*args, **kwargs)
                return [leaf, random_delta(leaf, alpha)]
        except Exception as e:
            print(func.__name__, str(e))
            raise e
    return _build_tree(depth=0)

def build_img_from_tree(tree, step=0):
    def build_img_(tree):
        if len(tree) == 2:
            leaf = tree[0] + tree[1] * step
            return leaf
        else:
            _, func, branches = tree
            args = [build_img_(branch) for branch in branches]
            return func(*args)
    return build_img_(tree)

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
    parser.add_option('-b', '--bitrate', dest='bitrate', type=str, default=None, help='Constant bitrate. Default: None (variable bitrate).')
    parser.add_option('-p', '--processes', dest='n_process', type=int, default=1, help='Number of parallel processes for frame creation. Default: 1.')
    args, _ = parser.parse_args()
    return args


def worker_fn(tree, step):
    img = build_img_from_tree(tree, step=step)
    img = np.rint(img.clip(0.0, 1.0)* 255.0).astype(np.uint8)
    Image.fromarray(img).save("frames/image-{}.png".format(str(step).rjust(5, "0")))

if __name__ == "__main__":

    args = parse_cmd_args()

    try:
        os.mkdir("videos")
    except:
        pass

    n_videos = args.n_videos if args.seed is None else 1

    start_i = max([int(x.split("-")[1].split(".")[0]) for x in os.listdir("videos")])

    for i, video_n in enumerate(rand(n_videos)):
        video_n = int(video_n * 1e9) if args.seed is None else args.seed
        try:
            os.mkdir("frames")
        except:
            shutil.rmtree("frames")
            time.sleep(0.5)
            os.mkdir("frames")

        duration = args.duration

        print(f"Creating frames for video with {args.fps} in {args.ext} format")

        tree = build_tree(min_depth=6, max_depth=15, seed=video_n, weights=p, dx=args.width, dy=args.height, alpha=4e-3)

        worker_args = [(tree, step) for step in range(args.fps*args.duration)]
        with mp.Pool(args.n_process) as pool:
            pool.starmap(worker_fn, worker_args)
        
        bitrate_arg = "" if args.bitrate is None else f"-b {args.bitrate}"

        os.system(f"ffmpeg -framerate {args.fps} -i frames/image-%05d.png {bitrate_arg} videos/video-{i+start_i}.{args.ext}")
