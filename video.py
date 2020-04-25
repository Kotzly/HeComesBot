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

try:
    os.mkdir("videos")
except:
    shutil.rmtree("videos")
    time.sleep(0.5)
    os.mkdir("videos")

def parse_cmd_args():
    parser = optparse.OptionParser()
    parser.add_option('-n', '--n_videos', dest='n_videos', type=int, default=1, help='Number of videos')
    parser.add_option('-f', '--fps', dest='fps', type=int, default=30, help='FPS.')
    parser.add_option('-H', '--height', dest='height', type=int, default=256, help='Video height.')
    parser.add_option('-W', '--width', dest='width', type=int, default=256, help='Video width.')
    parser.add_option('-s', '--step', dest='step', type=float, default=0.003, help='Alpha step for image generation.')
    parser.add_option('-d', '--duration', dest='duration', type=int, default=10, help='Video duration.')
    parser.add_option('-S', '--seed', dest='seed', type=int, default=None, help='Seed.')
    parser.add_option('-e', '--extension', dest='ext', type=str, default="webm", help='Extension. Can be webm, avi, mp4.')
    args, _ = parser.parse_args()
    return args

args = parse_cmd_args()
n_videos = args.n_videos if args.seed is None else 1

for i, video_n in enumerate(rand(n_videos)):
    video_n = int(video_n * 1e9) if args.seed is None else args.seed
    try:
        os.mkdir("frames")
    except:
        shutil.rmtree("frames")
        time.sleep(0.5)
        os.mkdir("frames")

    tree = build_tree(min_depth=6, max_depth=15, seed=video_n, weights=p, dx=args.width, dy=args.height, alpha=4e-3)

    def func(step):
        img = build_img_from_tree(tree, step=step)
        img = np.rint(img.clip(0.0, 1.0)* 255.0).astype(np.uint8)
        Image.fromarray(img).save("frames/image-{}.png".format(str(step).rjust(5, "0")))

    for step in range(args.fps*args.duration):
        func(step)

    os.system(f"ffmpeg -framerate {args.fps} -i frames/image-%05d.png videos/video-{i}.{args.ext}")
