import optparse
import pathlib

import numpy as np
from PIL import Image as PILImage

from hecomes.artgen.render import COLOR_SPACES, render_frame
from hecomes.artgen.tree import build_node, eval_node
from hecomes.config import PERSONALITIES_DIR, load_personality_list


def _parse_args():
    parser = optparse.OptionParser(usage="usage: %prog [options] [output.png]")
    parser.add_option("-s", "--seed", dest="seed", type=int, default=None,
                      help="Random seed. Default: random.")
    parser.add_option("-W", "--width", dest="width", type=int, default=512,
                      help="Image width. Default: 512.")
    parser.add_option("-H", "--height", dest="height", type=int, default=512,
                      help="Image height. Default: 512.")
    parser.add_option("--min-depth", dest="min_depth", type=int, default=6,
                      help="Minimum tree depth. Default: 6.")
    parser.add_option("--max-depth", dest="max_depth", type=int, default=12,
                      help="Maximum tree depth. Default: 12.")
    parser.add_option("--alpha", dest="alpha", type=float, default=4e-3,
                      help="Leaf delta scale. Default: 4e-3.")
    parser.add_option("-c", "--color-space", dest="color_space", type=str, default="rgb",
                      help=f"Color space: {', '.join(COLOR_SPACES)}. Default: rgb.")
    parser.add_option("--personality", dest="personality", type=str, default=None,
                      help="Personality name in data/personalities/. Default: personality.")
    args, positional = parser.parse_args()
    output = positional[0] if positional else "output.png"
    return args, output


def main():
    args, output_path = _parse_args()

    seed = args.seed if args.seed is not None else int(np.random.randint(0, 2**31))
    color_space = args.color_space if args.color_space in COLOR_SPACES else "rgb"

    personality = args.personality or "personality"
    weights = load_personality_list(PERSONALITIES_DIR / (personality + ".json"))

    np.random.seed(seed % (2**32 - 1))
    leaves = {}
    tree = build_node(0, args.min_depth, args.max_depth, args.width, args.height,
                      weights, args.alpha, leaves)

    steps = np.zeros((1, 1, 1, 1), dtype=np.float32)
    raw = eval_node(tree, steps, leaves)
    img_8 = render_frame(raw, color_space, args.width, args.height)

    output_path = pathlib.Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    PILImage.fromarray(img_8).save(output_path)
    print(f"Saved {args.width}x{args.height} image to {output_path} (seed={seed})")


if __name__ == "__main__":
    main()
