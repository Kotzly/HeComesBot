import optparse
import pathlib

import numpy as np
from PIL import Image as PILImage

from hecomesbot.artgen.functions import BUILD_FUNCTIONS, generate_params
from hecomesbot.artgen.render import COLOR_SPACES, render_frame
from hecomesbot.artgen.tree import get_random_function, random_delta
from hecomesbot.config import DATA_DIR, load_personality_list

_FUNC_BY_NAME = {f.__name__: (n, f) for n, f in BUILD_FUNCTIONS}


def _build_leaf(func, dx, dy, alpha):
    params = generate_params(func.__name__)
    base = func(dx=dx, dy=dy, **params).astype(np.float32)
    delta = np.float32(random_delta(alpha))
    return base, delta, params


def _build_tree(depth, min_depth, max_depth, dx, dy, weights, alpha, leaves):
    n_args, func = get_random_function(
        depth, p=weights, min_depth=min_depth, max_depth=max_depth
    )
    import uuid
    nid = str(uuid.uuid4())[:8]
    if n_args == 0:
        base, delta, params = _build_leaf(func, dx, dy, alpha)
        leaves[nid] = {"base": base, "delta": delta, "func": func.__name__, "params": params}
        return {"id": nid, "func": func.__name__, "arity": 0, "children": [],
                "delta": float(delta), "params": params}
    params = generate_params(func.__name__)
    children = [
        _build_tree(depth + 1, min_depth, max_depth, dx, dy, weights, alpha, leaves)
        for _ in range(n_args)
    ]
    return {"id": nid, "func": func.__name__, "arity": n_args, "children": children, "params": params}


def _eval_tree(node, steps, leaves):
    if node["arity"] == 0:
        leaf = leaves[node["id"]]
        return leaf["base"] + leaf["delta"] * steps
    _, func = _FUNC_BY_NAME[node["func"]]
    args = [_eval_tree(c, steps, leaves) for c in node["children"]]
    return func(*args, **node.get("params", {}))


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
    parser.add_option("--max-depth", dest="max_depth", type=int, default=16,
                      help="Maximum tree depth. Default: 16.")
    parser.add_option("--alpha", dest="alpha", type=float, default=4e-3,
                      help="Leaf delta scale. Default: 4e-3.")
    parser.add_option("-c", "--color-space", dest="color_space", type=str, default="rgb",
                      help=f"Color space: {', '.join(COLOR_SPACES)}. Default: rgb.")
    parser.add_option("--personality", dest="personality", type=str, default=None,
                      help="Personality JSON filename in data/. Default: personality.json.")
    args, positional = parser.parse_args()
    output = positional[0] if positional else "output.png"
    return args, output


def main():
    args, output_path = _parse_args()

    seed = args.seed if args.seed is not None else int(np.random.randint(0, 2**31))
    color_space = args.color_space if args.color_space in COLOR_SPACES else "rgb"

    personality_file = args.personality or "personality.json"
    weights = load_personality_list(DATA_DIR / personality_file)

    np.random.seed(seed % (2**32 - 1))
    leaves = {}
    tree = _build_tree(0, args.min_depth, args.max_depth, args.width, args.height,
                       weights, args.alpha, leaves)

    steps = np.zeros((1, 1, 1, 1), dtype=np.float32)
    raw = _eval_tree(tree, steps, leaves)
    img_8 = render_frame(raw, color_space, args.width, args.height)

    output_path = pathlib.Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    PILImage.fromarray(img_8).save(output_path)
    print(f"Saved {args.width}x{args.height} image to {output_path} (seed={seed})")


if __name__ == "__main__":
    main()
