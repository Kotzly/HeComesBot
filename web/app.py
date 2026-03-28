import base64
import colorsys
import io
import json as _json
import os
import pathlib
import pickle
import uuid

import numpy as np
from flask import Flask, jsonify, request, send_from_directory
from PIL import Image

from hecomes.artgen.functions import BUILD_FUNCTIONS, FUNC_PARAMS, generate_params
from hecomes.artgen.pruning import PRUNE_METHODS, image_entropy
from hecomes.artgen.render import COLOR_SPACES, render_frame
from hecomes.artgen.tree import get_random_function, random_delta
from hecomes.config import DATA_DIR, PERSONALITIES_DIR, load_personality_list

SAVE_DIR = pathlib.Path("saved_trees")

app = Flask(__name__, static_folder="static", static_url_path="")

_sessions = {}


FUNC_BY_NAME = {f.__name__: (n, f) for n, f in BUILD_FUNCTIONS}
FUNCS_BY_ARITY = {}
for _n, _f in BUILD_FUNCTIONS:
    FUNCS_BY_ARITY.setdefault(_n, []).append(_f.__name__)
for _arity in FUNCS_BY_ARITY:
    FUNCS_BY_ARITY[_arity].sort()


def _build_leaf(func, dx, dy, alpha):
    """Build a leaf array and capture editable params."""
    params = generate_params(func.__name__)
    base = func(dx=dx, dy=dy, **params).astype(np.float32)
    delta = np.float32(random_delta(alpha))
    return base, delta, params


def _recompute_leaf(func_name, params, dx, dy):
    """Recompute leaf base array from stored params."""
    _, f = FUNC_BY_NAME[func_name]
    return f(dx=dx, dy=dy, **params).astype(np.float32)


# ── Tree helpers ──────────────────────────────────────────────────────────────


def _new_id():
    return str(uuid.uuid4())[:8]


def _build_rich(depth, min_depth, max_depth, dx, dy, weights, alpha, leaves):
    n_args, func = get_random_function(
        depth, p=weights, min_depth=min_depth, max_depth=max_depth
    )
    nid = _new_id()
    if n_args == 0:
        base, delta, params = _build_leaf(func, dx, dy, alpha)
        leaves[nid] = {
            "base": base,
            "delta": delta,
            "func": func.__name__,
            "params": params,
        }
        return {
            "id": nid,
            "func": func.__name__,
            "arity": 0,
            "children": [],
            "delta": float(delta),
            "params": params,
        }
    params = generate_params(func.__name__)
    children = [
        _build_rich(depth + 1, min_depth, max_depth, dx, dy, weights, alpha, leaves)
        for _ in range(n_args)
    ]
    return {
        "id": nid,
        "func": func.__name__,
        "arity": n_args,
        "children": children,
        "params": params,
    }


def _eval_rich(node, steps, leaves):
    if node["arity"] == 0:
        leaf = leaves[node["id"]]
        return leaf["base"] + leaf["delta"] * steps
    _, func = FUNC_BY_NAME[node["func"]]
    args = [_eval_rich(c, steps, leaves) for c in node["children"]]
    return func(*args, **node.get("params", {}))


def _collect_leaf_ids(node):
    if node["arity"] == 0:
        return [node["id"]]
    ids = []
    for c in node["children"]:
        ids.extend(_collect_leaf_ids(c))
    return ids


def _find_node(tree, node_id):
    if tree["id"] == node_id:
        return tree
    for child in tree.get("children", []):
        found = _find_node(child, node_id)
        if found is not None:
            return found
    return None


def _find_parent(tree, node_id):
    for i, child in enumerate(tree.get("children", [])):
        if child["id"] == node_id:
            return tree, i
        result = _find_parent(child, node_id)
        if result is not None:
            return result
    return None


def _node_depth(tree, node_id, d=0):
    if tree["id"] == node_id:
        return d
    for child in tree.get("children", []):
        result = _node_depth(child, node_id, d + 1)
        if result is not None:
            return result
    return None


# ── Routes ────────────────────────────────────────────────────────────────────


@app.route("/")
def index():
    return send_from_directory("static", "index.html")


@app.route("/api/functions")
def get_functions():
    return jsonify({str(k): v for k, v in FUNCS_BY_ARITY.items()})


@app.route("/api/personalities")
def get_personalities():
    files = sorted(
        f[:-5] for f in os.listdir(PERSONALITIES_DIR) if f.endswith(".json")
    )
    return jsonify(files)


@app.route("/api/trees")
def list_trees():
    os.makedirs(SAVE_DIR, exist_ok=True)
    files = sorted(f[:-4] for f in os.listdir(SAVE_DIR) if f.endswith(".pkl"))
    return jsonify(files)


@app.route("/api/build", methods=["POST"])
def build():
    data = request.json
    seed = int(data.get("seed", 42))
    dx = int(data.get("width", 256))
    dy = int(data.get("height", 256))
    min_depth = int(data.get("min_depth", 4))
    max_depth = int(data.get("max_depth", 8))
    personality = data.get("personality", "personality")
    alpha = float(data.get("alpha", 4e-3))
    color_space = data.get("color_space", "rgb")
    if color_space not in COLOR_SPACES:
        color_space = "rgb"

    weights = load_personality_list(PERSONALITIES_DIR / (personality + ".json"))
    np.random.seed(seed % (2**32 - 1))
    leaves = {}
    tree = _build_rich(0, min_depth, max_depth, dx, dy, weights, alpha, leaves)

    tree_id = _new_id()
    _sessions[tree_id] = {
        "tree": tree,
        "leaves": leaves,
        "meta": {
            "dx": dx,
            "dy": dy,
            "seed": seed,
            "min_depth": min_depth,
            "max_depth": max_depth,
            "alpha": alpha,
            "color_space": color_space,
            "personality": personality,
        },
    }
    return jsonify(
        {"tree_id": tree_id, "tree": tree, "meta": _sessions[tree_id]["meta"]}
    )


@app.route("/api/preview", methods=["POST"])
def preview():
    data = request.json
    tree_id = data["tree_id"]
    session = _sessions.get(tree_id)
    if session is None:
        return jsonify({"error": "unknown tree_id"}), 404

    dx, dy = session["meta"]["dx"], session["meta"]["dy"]
    color_space = session["meta"].get("color_space", "rgb")
    steps = np.zeros((1, 1, 1, 1), dtype=np.float32)
    raw = _eval_rich(session["tree"], steps, session["leaves"])
    img_8 = render_frame(raw, color_space, dx, dy)
    buf = io.BytesIO()
    Image.fromarray(img_8).save(buf, format="PNG")
    b64 = base64.b64encode(buf.getvalue()).decode()
    return jsonify({"image": f"data:image/png;base64,{b64}"})


@app.route("/api/save", methods=["POST"])
def save_tree():
    data = request.json
    tree_id = data["tree_id"]
    name = data.get("name", tree_id).strip()
    name = "".join(c for c in name if c.isalnum() or c in "._- ")
    if not name:
        return jsonify({"error": "invalid name"}), 400

    session = _sessions.get(tree_id)
    if session is None:
        return jsonify({"error": "unknown tree_id"}), 404

    os.makedirs(SAVE_DIR, exist_ok=True)
    path = os.path.join(SAVE_DIR, f"{name}.pkl")
    with open(path, "wb") as f:
        pickle.dump(session, f)
    return jsonify({"name": name})


@app.route("/api/load", methods=["POST"])
def load_tree():
    data = request.json
    name = os.path.basename(data.get("name", ""))
    path = os.path.join(SAVE_DIR, f"{name}.pkl")
    if not os.path.exists(path):
        return jsonify({"error": "file not found"}), 404

    with open(path, "rb") as f:
        session = pickle.load(f)

    tree_id = _new_id()
    _sessions[tree_id] = session
    return jsonify(
        {"tree_id": tree_id, "tree": session["tree"], "meta": session["meta"]}
    )


@app.route("/api/node/preview", methods=["POST"])
def node_preview():
    data = request.json
    tree_id = data["tree_id"]
    node_id = data["node_id"]

    session = _sessions.get(tree_id)
    if session is None:
        return jsonify({"error": "unknown tree_id"}), 404

    node = _find_node(session["tree"], node_id)
    if node is None:
        return jsonify({"error": "unknown node_id"}), 404

    dx, dy = session["meta"]["dx"], session["meta"]["dy"]
    color_space = session["meta"].get("color_space", "rgb")
    steps = np.zeros((1, 1, 1, 1), dtype=np.float32)
    raw = _eval_rich(node, steps, session["leaves"])
    img_8 = render_frame(raw, color_space, dx, dy)
    buf = io.BytesIO()
    Image.fromarray(img_8).save(buf, format="PNG")
    b64 = base64.b64encode(buf.getvalue()).decode()
    return jsonify({"image": f"data:image/png;base64,{b64}"})


@app.route("/api/node/set-func", methods=["POST"])
def set_func():
    data = request.json
    tree_id = data["tree_id"]
    node_id = data["node_id"]
    func_name = data["func_name"]

    session = _sessions.get(tree_id)
    if session is None:
        return jsonify({"error": "unknown tree_id"}), 404
    if func_name not in FUNC_BY_NAME:
        return jsonify({"error": f"unknown function: {func_name}"}), 400

    new_arity, new_func = FUNC_BY_NAME[func_name]
    node = _find_node(session["tree"], node_id)
    if node is None:
        return jsonify({"error": "unknown node_id"}), 404

    old_arity = node["arity"]
    meta = session["meta"]

    if new_arity == old_arity:
        node["func"] = func_name
        # If becoming a leaf with same arity (0→0), regenerate leaf data
        if new_arity == 0:
            for lid in _collect_leaf_ids(node):
                session["leaves"].pop(lid, None)
            base, delta, params = _build_leaf(
                new_func, meta["dx"], meta["dy"], meta["alpha"]
            )
            session["leaves"][node_id] = {
                "base": base,
                "delta": delta,
                "func": func_name,
                "params": params,
            }
            node.update({"delta": float(delta), "params": params})
        else:
            node["params"] = generate_params(func_name)
        return jsonify({"tree": session["tree"]})

    # Arity is changing
    for lid in _collect_leaf_ids(node):
        session["leaves"].pop(lid, None)

    node["func"] = func_name
    node["arity"] = new_arity

    if new_arity == 0:
        base, delta, params = _build_leaf(
            new_func, meta["dx"], meta["dy"], meta["alpha"]
        )
        session["leaves"][node_id] = {
            "base": base,
            "delta": delta,
            "func": func_name,
            "params": params,
        }
        node.update({"children": [], "delta": float(delta), "params": params})
    else:
        nd = _node_depth(session["tree"], node_id) or 0
        child_depth = nd + 1
        eff_max = max(meta["max_depth"] - child_depth, 2)
        eff_min = max(min(meta["min_depth"] - child_depth, eff_max - 1), 1)
        weights = load_personality_list(PERSONALITIES_DIR / (meta.get("personality", "personality") + ".json"))
        new_children = []
        for _ in range(new_arity):
            np.random.seed(np.random.randint(0, 2**31))
            child = _build_rich(
                0,
                eff_min,
                eff_max,
                meta["dx"],
                meta["dy"],
                weights,
                meta["alpha"],
                session["leaves"],
            )
            new_children.append(child)
        node["children"] = new_children
        node["params"] = generate_params(func_name)
        node.pop("delta", None)

    return jsonify({"tree": session["tree"]})


@app.route("/api/node/regenerate", methods=["POST"])
def regenerate():
    data = request.json
    tree_id = data["tree_id"]
    node_id = data["node_id"]
    seed = int(data.get("seed", np.random.randint(0, 2**31)))

    session = _sessions.get(tree_id)
    if session is None:
        return jsonify({"error": "unknown tree_id"}), 404

    node = _find_node(session["tree"], node_id)
    if node is None:
        return jsonify({"error": "unknown node_id"}), 404

    meta = session["meta"]
    for lid in _collect_leaf_ids(node):
        session["leaves"].pop(lid, None)

    nd = _node_depth(session["tree"], node_id) or 0
    eff_max = max(meta["max_depth"] - nd, 2)
    eff_min = max(min(meta["min_depth"] - nd, eff_max - 1), 1)

    weights = load_personality_list(DATA_DIR / "personality.json")
    np.random.seed(seed % (2**32 - 1))
    new_subtree = _build_rich(
        0,
        eff_min,
        eff_max,
        meta["dx"],
        meta["dy"],
        weights,
        meta["alpha"],
        session["leaves"],
    )

    if session["tree"]["id"] == node_id:
        session["tree"] = new_subtree
    else:
        parent, idx = _find_parent(session["tree"], node_id)
        parent["children"][idx] = new_subtree

    return jsonify({"tree": session["tree"], "new_node_id": new_subtree["id"]})


@app.route("/api/leaf/set-params", methods=["POST"])
def set_leaf_params():
    data = request.json
    tree_id = data["tree_id"]
    node_id = data["node_id"]
    new_params = data.get("params", {})
    new_delta = data.get("delta", None)

    session = _sessions.get(tree_id)
    if session is None:
        return jsonify({"error": "unknown tree_id"}), 404

    node = _find_node(session["tree"], node_id)
    if node is None or node["arity"] != 0:
        return jsonify({"error": "node not found or not a leaf"}), 404

    leaf = session["leaves"].get(node_id)
    if leaf is None:
        return jsonify({"error": "leaf data missing"}), 500

    meta = session["meta"]
    _push_undo(session)

    if new_params:
        merged = {**leaf.get("params", {}), **new_params}
        new_base = _recompute_leaf(node["func"], merged, meta["dx"], meta["dy"])
        if new_base is not None:
            leaf["base"] = new_base
            leaf["params"] = merged
            node["params"] = merged

    if new_delta is not None:
        leaf["delta"] = np.float32(new_delta)
        node["delta"] = float(new_delta)

    return jsonify({"tree": session["tree"]})


@app.route("/api/function-params")
def get_function_params():
    return jsonify(FUNC_PARAMS)


@app.route("/api/node/set-params", methods=["POST"])
def set_node_params():
    data = request.json
    tree_id = data["tree_id"]
    node_id = data["node_id"]
    new_params = data.get("params", {})

    session = _sessions.get(tree_id)
    if session is None:
        return jsonify({"error": "unknown tree_id"}), 404

    node = _find_node(session["tree"], node_id)
    if node is None or node["arity"] == 0:
        return jsonify({"error": "node not found or is a leaf"}), 404

    node["params"] = {**node.get("params", {}), **new_params}
    return jsonify({"tree": session["tree"]})


@app.route("/api/node/flatten", methods=["POST"])
def flatten_node():
    data = request.json
    tree_id = data["tree_id"]
    node_id = data["node_id"]

    session = _sessions.get(tree_id)
    if session is None:
        return jsonify({"error": "unknown tree_id"}), 404

    node = _find_node(session["tree"], node_id)
    if node is None:
        return jsonify({"error": "unknown node_id"}), 404
    if node["arity"] == 0:
        return jsonify({"error": "node is already a leaf"}), 400

    meta = session["meta"]
    dx, dy = meta["dx"], meta["dy"]
    color_space = meta.get("color_space", "rgb")
    alpha = meta.get("alpha", 4e-3)
    _push_undo(session)

    steps = np.zeros((1, 1, 1, 1), dtype=np.float32)
    raw = _eval_rich(node, steps, session["leaves"])
    frame = render_frame(raw, color_space, dx, dy)
    mean_rgb = frame.mean(axis=(0, 1)) / 255.0
    mean_color = _rgb_to_color_space(mean_rgb, color_space)

    for lid in _collect_leaf_ids(node):
        session["leaves"].pop(lid, None)

    new_node, new_leaf = _make_rand_color_leaf(mean_color, dx, dy, alpha)

    if session["tree"]["id"] == node_id:
        session["tree"] = new_node
    else:
        parent, idx = _find_parent(session["tree"], node_id)
        parent["children"][idx] = new_node

    session["leaves"][new_node["id"]] = new_leaf
    return jsonify({"tree": session["tree"], "new_node_id": new_node["id"]})


_UNDO_LIMIT = 20


def _push_undo(session):
    stack = session.setdefault("_undo_stack", [])
    stack.append({
        "tree":   _json.loads(_json.dumps(session["tree"])),
        "leaves": {k: {**v, "base": v["base"].copy()} for k, v in session["leaves"].items()},
    })
    if len(stack) > _UNDO_LIMIT:
        stack.pop(0)


def _make_rand_color_leaf(mean_color, dx, dy, alpha):
    nid = _new_id()
    params = {"color": [float(mean_color[i]) for i in range(3)]}
    _, rand_color_fn = FUNC_BY_NAME["rand_color"]
    base = rand_color_fn(dx=dx, dy=dy, **params).astype(np.float32)
    delta = np.float32(alpha)
    node = {
        "id": nid, "func": "rand_color", "arity": 0,
        "children": [], "delta": float(delta), "params": params,
    }
    leaf_data = {"base": base, "delta": delta, "func": "rand_color", "params": params}
    return node, leaf_data


def _rgb_to_color_space(mean_rgb, color_space):
    r, g, b = float(mean_rgb[0]), float(mean_rgb[1]), float(mean_rgb[2])
    if color_space == "hsv":
        return list(colorsys.rgb_to_hsv(r, g, b))
    if color_space == "cmy":
        return [1.0 - r, 1.0 - g, 1.0 - b]
    return [r, g, b]


def _collect_parents_postorder(node, result):
    """Collect all non-leaf nodes in post-order (bottom-up)."""
    for child in node.get("children", []):
        _collect_parents_postorder(child, result)
    if node.get("children"):
        result.append(node)


def _prune_pass(subtree_root, leaves, dx, dy, color_space, method_fn, delta, threshold, alpha):
    """One bottom-up pruning pass. Returns number of subbranches pruned."""
    steps = np.zeros((1, 1, 1, 1), dtype=np.float32)
    _TEMP_ID = "__prune_temp__"

    original_raw = _eval_rich(subtree_root, steps, leaves)
    original_frame = render_frame(original_raw, color_space, dx, dy)

    parents = []
    _collect_parents_postorder(subtree_root, parents)

    n_pruned = 0
    for parent in parents:
        for i, child in enumerate(parent["children"]):
            if child["func"] == "rand_color":
                continue

            # Evaluate the child subbranch
            child_raw = _eval_rich(child, steps, leaves)
            child_base = child_raw[0] if child_raw.ndim == 4 else child_raw

            # Temporarily replace child with a perturbed constant node
            leaves[_TEMP_ID] = {
                "base": (child_base + np.float32(delta)).astype(np.float32),
                "delta": np.float32(0), "func": "rand_color", "params": {},
            }
            temp_node = {"id": _TEMP_ID, "func": "rand_color", "arity": 0, "children": [], "delta": 0.0, "params": {}}
            parent["children"][i] = temp_node

            perturbed_raw = _eval_rich(subtree_root, steps, leaves)
            perturbed_frame = render_frame(perturbed_raw, color_space, dx, dy)

            # Restore
            parent["children"][i] = child
            del leaves[_TEMP_ID]

            if method_fn(original_frame, perturbed_frame, threshold=threshold):
                # Mean color from the rendered child output (visually accurate)
                child_frame = render_frame(child_raw, color_space, dx, dy)
                mean_rgb = child_frame.mean(axis=(0, 1)) / 255.0
                mean_color = _rgb_to_color_space(mean_rgb, color_space)

                for lid in _collect_leaf_ids(child):
                    leaves.pop(lid, None)

                new_node, new_leaf_data = _make_rand_color_leaf(mean_color, dx, dy, alpha)
                parent["children"][i] = new_node
                leaves[new_node["id"]] = new_leaf_data
                n_pruned += 1

    return n_pruned


def _compute_sensitivity(tree_root, leaves, dx, dy, color_space, delta):
    """For every non-leaf node N compute:
      - root:  |ΔH| at tree root when N's output is perturbed
      - leaf:  mean |ΔH| at N when each of its leaf descendants is perturbed
    Returns {node_id: {"root": float, "leaf": float}}
    """
    steps = np.zeros((1, 1, 1, 1), dtype=np.float32)
    temp_id = "__sens_temp__"
    result = {}

    original_root_raw = _eval_rich(tree_root, steps, leaves)
    original_root_frame = render_frame(original_root_raw, color_space, dx, dy)
    original_root_entropy = image_entropy(original_root_frame)

    def visit(node):
        if node["arity"] == 0:
            return

        # ── Root sensitivity ──────────────────────────────────────────────
        node_raw = _eval_rich(node, steps, leaves)
        node_base = node_raw[0] if node_raw.ndim == 4 else node_raw

        if tree_root["id"] == node["id"]:
            root_sens = 0.0
        else:
            parent, idx = _find_parent(tree_root, node["id"])
            leaves[temp_id] = {
                "base": (node_base + np.float32(delta)).astype(np.float32),
                "delta": np.float32(0), "func": "rand_color", "params": {},
            }
            temp_node = {"id": temp_id, "func": "rand_color", "arity": 0,
                         "children": [], "delta": 0.0, "params": {}}
            parent["children"][idx] = temp_node

            perturbed_root_frame = render_frame(
                _eval_rich(tree_root, steps, leaves), color_space, dx, dy
            )
            parent["children"][idx] = node
            del leaves[temp_id]

            root_sens = round(abs(image_entropy(perturbed_root_frame) - original_root_entropy), 4)

        # ── Avg leaf sensitivity ──────────────────────────────────────────
        original_node_entropy = image_entropy(
            render_frame(_eval_rich(node, steps, leaves), color_space, dx, dy)
        )
        leaf_deltas = []
        for lid in _collect_leaf_ids(node):
            if lid not in leaves:
                continue
            leaf = leaves[lid]
            original_base = leaf["base"]
            leaf["base"] = original_base + np.float32(delta)

            perturbed_node_entropy = image_entropy(
                render_frame(_eval_rich(node, steps, leaves), color_space, dx, dy)
            )
            leaf["base"] = original_base
            leaf_deltas.append(abs(perturbed_node_entropy - original_node_entropy))

        avg_leaf = round(float(np.mean(leaf_deltas)) if leaf_deltas else 0.0, 4)
        result[node["id"]] = {"root": root_sens, "leaf": avg_leaf}

        for child in node.get("children", []):
            visit(child)

    visit(tree_root)
    return result


@app.route("/api/sensitivity", methods=["POST"])
def sensitivity():
    data = request.json
    tree_id = data["tree_id"]
    delta = float(data.get("delta", 0.05))

    session = _sessions.get(tree_id)
    if session is None:
        return jsonify({"error": "unknown tree_id"}), 404

    meta = session["meta"]
    dx, dy = meta["dx"], meta["dy"]
    color_space = meta.get("color_space", "rgb")

    reference_node_id = data.get("reference_node_id")

    tree_copy = _json.loads(_json.dumps(session["tree"]))
    leaves_copy = {k: {**v, "base": v["base"].copy()} for k, v in session["leaves"].items()}

    if reference_node_id:
        root_copy = _find_node(tree_copy, reference_node_id)
        if root_copy is None:
            return jsonify({"error": "unknown reference_node_id"}), 404
    else:
        root_copy = tree_copy

    return jsonify(_compute_sensitivity(root_copy, leaves_copy, dx, dy, color_space, delta))


@app.route("/api/undo", methods=["POST"])
def undo():
    data = request.json
    session = _sessions.get(data["tree_id"])
    if session is None:
        return jsonify({"error": "unknown tree_id"}), 404
    stack = session.get("_undo_stack", [])
    if not stack:
        return jsonify({"error": "nothing to undo"}), 400
    snapshot = stack.pop()
    session["tree"]   = snapshot["tree"]
    session["leaves"] = snapshot["leaves"]
    return jsonify({"tree": session["tree"]})


@app.route("/api/prune-methods")
def get_prune_methods():
    return jsonify(list(PRUNE_METHODS.keys()))


@app.route("/api/prune", methods=["POST"])
def prune():
    data = request.json
    tree_id = data["tree_id"]
    node_id = data["node_id"]
    method_name = data.get("method", "entropy_sensitivity")
    delta = float(data.get("delta", 0.05))
    threshold = float(data.get("threshold", 0.1))

    session = _sessions.get(tree_id)
    if session is None:
        return jsonify({"error": "unknown tree_id"}), 404

    node = _find_node(session["tree"], node_id)
    if node is None:
        return jsonify({"error": "unknown node_id"}), 404
    if node["arity"] == 0:
        return jsonify({"error": "cannot prune a leaf node"}), 400

    method_fn = PRUNE_METHODS.get(method_name)
    if method_fn is None:
        return jsonify({"error": f"unknown prune method: {method_name}"}), 400

    meta = session["meta"]
    dx, dy = meta["dx"], meta["dy"]
    color_space = meta.get("color_space", "rgb")
    alpha = meta.get("alpha", 4e-3)

    _push_undo(session)
    total_pruned = 0
    while True:
        n = _prune_pass(node, session["leaves"], dx, dy, color_space, method_fn, delta, threshold, alpha)
        total_pruned += n
        if n == 0:
            break

    return jsonify({"tree": session["tree"], "pruned": total_pruned})


def main():
    app.run(debug=True, port=5000, host="0.0.0.0")


if __name__ == "__main__":
    main()
