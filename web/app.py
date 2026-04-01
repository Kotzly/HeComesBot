import base64
import colorsys
import io
import os
import pathlib
import pickle
import uuid

import numpy as np
from flask import Flask, jsonify, request, send_from_directory
from PIL import Image

from hecomes.artgen.functions import FUNCTION_REGISTRY, FUNC_PARAMS, REGISTRY_BY_NAME
from hecomes.artgen.pruning import PRUNE_METHODS, image_entropy
from hecomes.artgen.render import COLOR_SPACES, render_frame
from hecomes.artgen.tree import (
    Node,
    build_node,
    collect_leaf_ids,
    eval_node,
    find_parent,
    node_depth,
    nodes_from_dict,
    nodes_to_dict,
    random_delta,
)
from hecomes.config import PERSONALITIES_DIR, load_personality_list

SAVE_DIR = pathlib.Path("saved_trees")

app = Flask(__name__, static_folder="static", static_url_path="")

_sessions = {}

FUNCS_BY_ARITY = {}
for _fd in FUNCTION_REGISTRY:
    FUNCS_BY_ARITY.setdefault(_fd.arity, []).append(_fd.func.__name__)
for _arity in FUNCS_BY_ARITY:
    FUNCS_BY_ARITY[_arity].sort()


# ── Session helpers ────────────────────────────────────────────────────────────


def _new_id():
    return str(uuid.uuid4())[:8]


def _tree_response(session) -> dict:
    return {"root_id": session["root_id"], "nodes": nodes_to_dict(session["nodes"])}


def _make_leaf_node(fd, dx, dy, alpha) -> tuple[Node, np.ndarray]:
    params = fd.generate() if fd.generate else {}
    base = fd.func(dx=dx, dy=dy, **params).astype(np.float32)
    node = Node(func=fd, params=params, delta=float(random_delta(alpha)))
    return node, base


def _make_rand_color_leaf(mean_color, dx, dy, alpha) -> tuple[Node, np.ndarray]:
    fd = REGISTRY_BY_NAME["rand_color"]
    params = {"color": [float(mean_color[i]) for i in range(3)]}
    base = fd.func(dx=dx, dy=dy, **params).astype(np.float32)
    node = Node(func=fd, params=params, delta=float(alpha))
    return node, base


def _remove_subtree(node_id: str, nodes: dict):
    """Remove node and all its descendants from the nodes dict."""
    node = nodes.pop(node_id, None)
    if node:
        for cid in node.children:
            _remove_subtree(cid, nodes)


# ── Undo ──────────────────────────────────────────────────────────────────────

_UNDO_LIMIT = 20


def _push_undo(session):
    stack = session.setdefault("_undo_stack", [])
    stack.append({
        "root_id": session["root_id"],
        "nodes": nodes_to_dict(session["nodes"]),
        "leaves": {k: v.copy() for k, v in session["leaves"].items()},
    })
    if len(stack) > _UNDO_LIMIT:
        stack.pop(0)


# ── Color helpers ─────────────────────────────────────────────────────────────


def _rgb_to_color_space(mean_rgb, color_space):
    r, g, b = float(mean_rgb[0]), float(mean_rgb[1]), float(mean_rgb[2])
    if color_space == "hsv":
        return list(colorsys.rgb_to_hsv(r, g, b))
    if color_space == "cmy":
        return [1.0 - r, 1.0 - g, 1.0 - b]
    return [r, g, b]


# ── Pruning helpers ───────────────────────────────────────────────────────────


def _collect_parents_postorder(node_id: str, nodes: dict, result: list):
    """Collect non-leaf node IDs in post-order (children before parents)."""
    node = nodes[node_id]
    for cid in node.children:
        _collect_parents_postorder(cid, nodes, result)
    if node.children:
        result.append(node_id)


_TEMP_ID = "__prune_temp__"


def _prune_pass(subtree_root_id, nodes, leaves, dx, dy, color_space, method_fn, delta, threshold, alpha):
    steps = np.zeros((1, 1, 1, 1), dtype=np.float32)
    rand_color_fd = REGISTRY_BY_NAME["rand_color"]

    original_raw = eval_node(subtree_root_id, nodes, leaves, steps)
    original_frame = render_frame(original_raw, color_space, dx, dy)

    parents_postorder = []
    _collect_parents_postorder(subtree_root_id, nodes, parents_postorder)

    n_pruned = 0
    for parent_id in parents_postorder:
        parent_node = nodes.get(parent_id)
        if parent_node is None:
            continue
        for i, child_id in enumerate(list(parent_node.children)):
            child_node = nodes.get(child_id)
            if child_node is None or child_node.func.func.__name__ == "rand_color":
                continue

            child_raw = eval_node(child_id, nodes, leaves, steps)
            child_base = child_raw[0] if child_raw.ndim == 4 else child_raw

            leaves[_TEMP_ID] = (child_base + np.float32(delta)).astype(np.float32)
            nodes[_TEMP_ID] = Node(func=rand_color_fd, id=_TEMP_ID, delta=0.0)
            parent_node.children[i] = _TEMP_ID

            perturbed_raw = eval_node(subtree_root_id, nodes, leaves, steps)
            perturbed_frame = render_frame(perturbed_raw, color_space, dx, dy)

            parent_node.children[i] = child_id
            del nodes[_TEMP_ID]
            del leaves[_TEMP_ID]

            if method_fn(original_frame, perturbed_frame, threshold=threshold):
                child_frame = render_frame(eval_node(child_id, nodes, leaves, steps), color_space, dx, dy)
                mean_rgb = child_frame.mean(axis=(0, 1)) / 255.0
                mean_color = _rgb_to_color_space(mean_rgb, color_space)

                for lid in collect_leaf_ids(child_id, nodes):
                    leaves.pop(lid, None)
                _remove_subtree(child_id, nodes)

                new_node, new_base = _make_rand_color_leaf(mean_color, dx, dy, alpha)
                nodes[new_node.id] = new_node
                leaves[new_node.id] = new_base
                parent_node.children[i] = new_node.id
                n_pruned += 1

    return n_pruned


def _compute_sensitivity(root_id, nodes, leaves, dx, dy, color_space, delta):
    steps = np.zeros((1, 1, 1, 1), dtype=np.float32)
    rand_color_fd = REGISTRY_BY_NAME["rand_color"]
    result = {}

    original_root_raw = eval_node(root_id, nodes, leaves, steps)
    original_root_entropy = image_entropy(render_frame(original_root_raw, color_space, dx, dy))

    def visit(node_id):
        node = nodes[node_id]
        if node.arity == 0:
            return

        node_raw = eval_node(node_id, nodes, leaves, steps)
        node_base = node_raw[0] if node_raw.ndim == 4 else node_raw

        if root_id == node_id:
            root_sens = 0.0
        else:
            parent_id, idx = find_parent(nodes, node_id)
            parent_node = nodes[parent_id]
            leaves[_TEMP_ID] = (node_base + np.float32(delta)).astype(np.float32)
            nodes[_TEMP_ID] = Node(func=rand_color_fd, id=_TEMP_ID, delta=0.0)
            parent_node.children[idx] = _TEMP_ID

            perturbed_root_frame = render_frame(
                eval_node(root_id, nodes, leaves, steps), color_space, dx, dy
            )
            parent_node.children[idx] = node_id
            del nodes[_TEMP_ID]
            del leaves[_TEMP_ID]

            root_sens = round(abs(image_entropy(perturbed_root_frame) - original_root_entropy), 4)

        original_node_entropy = image_entropy(
            render_frame(eval_node(node_id, nodes, leaves, steps), color_space, dx, dy)
        )
        leaf_deltas = []
        for lid in collect_leaf_ids(node_id, nodes):
            if lid not in leaves:
                continue
            original_base = leaves[lid]
            leaves[lid] = original_base + np.float32(delta)
            perturbed_entropy = image_entropy(
                render_frame(eval_node(node_id, nodes, leaves, steps), color_space, dx, dy)
            )
            leaves[lid] = original_base
            leaf_deltas.append(abs(perturbed_entropy - original_node_entropy))

        avg_leaf = round(float(np.mean(leaf_deltas)) if leaf_deltas else 0.0, 4)
        result[node_id] = {"root": root_sens, "leaf": avg_leaf}

        for cid in node.children:
            visit(cid)

    visit(root_id)
    return result


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
    nodes, leaves = {}, {}
    root_id = build_node(0, min_depth, max_depth, dx, dy, weights, alpha, nodes, leaves)

    tree_id = _new_id()
    meta = {"dx": dx, "dy": dy, "seed": seed, "min_depth": min_depth,
            "max_depth": max_depth, "alpha": alpha, "color_space": color_space,
            "personality": personality}
    _sessions[tree_id] = {"root_id": root_id, "nodes": nodes, "leaves": leaves, "meta": meta}
    return jsonify({"tree_id": tree_id, **_tree_response(_sessions[tree_id]), "meta": meta})


@app.route("/api/preview", methods=["POST"])
def preview():
    data = request.json
    session = _sessions.get(data["tree_id"])
    if session is None:
        return jsonify({"error": "unknown tree_id"}), 404

    dx, dy = session["meta"]["dx"], session["meta"]["dy"]
    color_space = session["meta"].get("color_space", "rgb")
    steps = np.zeros((1, 1, 1, 1), dtype=np.float32)
    raw = eval_node(session["root_id"], session["nodes"], session["leaves"], steps)
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
    with open(os.path.join(SAVE_DIR, f"{name}.pkl"), "wb") as f:
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
    return jsonify({"tree_id": tree_id, **_tree_response(session), "meta": session["meta"]})


@app.route("/api/node/preview", methods=["POST"])
def node_preview():
    data = request.json
    tree_id = data["tree_id"]
    node_id = data["node_id"]

    session = _sessions.get(tree_id)
    if session is None:
        return jsonify({"error": "unknown tree_id"}), 404
    if node_id not in session["nodes"]:
        return jsonify({"error": "unknown node_id"}), 404

    dx, dy = session["meta"]["dx"], session["meta"]["dy"]
    color_space = session["meta"].get("color_space", "rgb")
    steps = np.zeros((1, 1, 1, 1), dtype=np.float32)
    raw = eval_node(node_id, session["nodes"], session["leaves"], steps)
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

    new_fd = REGISTRY_BY_NAME.get(func_name)
    if new_fd is None:
        return jsonify({"error": f"unknown function: {func_name}"}), 400

    node = session["nodes"].get(node_id)
    if node is None:
        return jsonify({"error": "unknown node_id"}), 404

    meta = session["meta"]
    old_arity = node.arity
    new_arity = new_fd.arity

    if new_arity == old_arity:
        node.func = new_fd
        if new_arity == 0:
            params = new_fd.generate() if new_fd.generate else {}
            base = new_fd.func(dx=meta["dx"], dy=meta["dy"], **params).astype(np.float32)
            session["leaves"][node_id] = base
            node.params = params
            node.delta = float(random_delta(meta["alpha"]))
        else:
            node.params = new_fd.generate() if new_fd.generate else {}
        return jsonify(_tree_response(session))

    # Arity is changing — remove old children / leaf
    for cid in node.children:
        for lid in collect_leaf_ids(cid, session["nodes"]):
            session["leaves"].pop(lid, None)
        _remove_subtree(cid, session["nodes"])

    if new_arity == 0:
        session["leaves"].pop(node_id, None)
        params = new_fd.generate() if new_fd.generate else {}
        base = new_fd.func(dx=meta["dx"], dy=meta["dy"], **params).astype(np.float32)
        session["leaves"][node_id] = base
        node.func = new_fd
        node.params = params
        node.delta = float(random_delta(meta["alpha"]))
        node.children = []
    else:
        session["leaves"].pop(node_id, None)
        nd = node_depth(session["nodes"], session["root_id"], node_id) or 0
        eff_max = max(meta["max_depth"] - nd - 1, 2)
        eff_min = max(min(meta["min_depth"] - nd - 1, eff_max - 1), 1)
        weights = load_personality_list(PERSONALITIES_DIR / (meta.get("personality", "personality") + ".json"))
        new_children = []
        for _ in range(new_arity):
            np.random.seed(np.random.randint(0, 2**31))
            child_id = build_node(0, eff_min, eff_max, meta["dx"], meta["dy"],
                                  weights, meta["alpha"], session["nodes"], session["leaves"])
            new_children.append(child_id)
        node.func = new_fd
        node.params = new_fd.generate() if new_fd.generate else {}
        node.delta = None
        node.children = new_children

    return jsonify(_tree_response(session))


@app.route("/api/node/regenerate", methods=["POST"])
def regenerate():
    data = request.json
    tree_id = data["tree_id"]
    node_id = data["node_id"]
    seed = int(data.get("seed", np.random.randint(0, 2**31)))

    session = _sessions.get(tree_id)
    if session is None:
        return jsonify({"error": "unknown tree_id"}), 404
    if node_id not in session["nodes"]:
        return jsonify({"error": "unknown node_id"}), 404

    meta = session["meta"]
    nd = node_depth(session["nodes"], session["root_id"], node_id) or 0
    eff_max = max(meta["max_depth"] - nd, 2)
    eff_min = max(min(meta["min_depth"] - nd, eff_max - 1), 1)

    parent_result = None
    if session["root_id"] != node_id:
        parent_result = find_parent(session["nodes"], node_id)

    for lid in collect_leaf_ids(node_id, session["nodes"]):
        session["leaves"].pop(lid, None)
    _remove_subtree(node_id, session["nodes"])

    weights = load_personality_list(PERSONALITIES_DIR / (meta.get("personality", "personality") + ".json"))
    np.random.seed(seed % (2**32 - 1))
    new_root_id = build_node(0, eff_min, eff_max, meta["dx"], meta["dy"],
                             weights, meta["alpha"], session["nodes"], session["leaves"])

    if parent_result:
        parent_id, idx = parent_result
        session["nodes"][parent_id].children[idx] = new_root_id
    else:
        session["root_id"] = new_root_id

    return jsonify({**_tree_response(session), "new_node_id": new_root_id})


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

    node = session["nodes"].get(node_id)
    if node is None or node.arity != 0:
        return jsonify({"error": "node not found or not a leaf"}), 404
    if node_id not in session["leaves"]:
        return jsonify({"error": "leaf data missing"}), 500

    meta = session["meta"]
    _push_undo(session)

    if new_params:
        merged = {**node.params, **new_params}
        node.params = merged
        session["leaves"][node_id] = node.func.func(
            dx=meta["dx"], dy=meta["dy"], **merged
        ).astype(np.float32)

    if new_delta is not None:
        node.delta = float(new_delta)

    return jsonify(_tree_response(session))


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

    node = session["nodes"].get(node_id)
    if node is None or node.arity == 0:
        return jsonify({"error": "node not found or is a leaf"}), 404

    node.params = {**node.params, **new_params}
    return jsonify(_tree_response(session))


@app.route("/api/node/flatten", methods=["POST"])
def flatten_node():
    data = request.json
    tree_id = data["tree_id"]
    node_id = data["node_id"]

    session = _sessions.get(tree_id)
    if session is None:
        return jsonify({"error": "unknown tree_id"}), 404

    node = session["nodes"].get(node_id)
    if node is None:
        return jsonify({"error": "unknown node_id"}), 404
    if node.arity == 0:
        return jsonify({"error": "node is already a leaf"}), 400

    meta = session["meta"]
    dx, dy = meta["dx"], meta["dy"]
    color_space = meta.get("color_space", "rgb")
    alpha = meta.get("alpha", 4e-3)
    _push_undo(session)

    parent_result = None
    if session["root_id"] != node_id:
        parent_result = find_parent(session["nodes"], node_id)

    steps = np.zeros((1, 1, 1, 1), dtype=np.float32)
    raw = eval_node(node_id, session["nodes"], session["leaves"], steps)
    frame = render_frame(raw, color_space, dx, dy)
    mean_rgb = frame.mean(axis=(0, 1)) / 255.0
    mean_color = _rgb_to_color_space(mean_rgb, color_space)

    for lid in collect_leaf_ids(node_id, session["nodes"]):
        session["leaves"].pop(lid, None)
    _remove_subtree(node_id, session["nodes"])

    new_node, new_base = _make_rand_color_leaf(mean_color, dx, dy, alpha)
    session["nodes"][new_node.id] = new_node
    session["leaves"][new_node.id] = new_base

    if parent_result:
        parent_id, idx = parent_result
        session["nodes"][parent_id].children[idx] = new_node.id
    else:
        session["root_id"] = new_node.id

    return jsonify({**_tree_response(session), "new_node_id": new_node.id})


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

    nodes_copy = nodes_from_dict(nodes_to_dict(session["nodes"]))
    leaves_copy = {k: v.copy() for k, v in session["leaves"].items()}

    root_id = reference_node_id if reference_node_id else session["root_id"]
    if root_id not in nodes_copy:
        return jsonify({"error": "unknown reference_node_id"}), 404

    return jsonify(_compute_sensitivity(root_id, nodes_copy, leaves_copy, dx, dy, color_space, delta))


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
    session["root_id"] = snapshot["root_id"]
    session["nodes"] = nodes_from_dict(snapshot["nodes"])
    session["leaves"] = snapshot["leaves"]
    return jsonify(_tree_response(session))


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

    node = session["nodes"].get(node_id)
    if node is None:
        return jsonify({"error": "unknown node_id"}), 404
    if node.arity == 0:
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
        n = _prune_pass(node_id, session["nodes"], session["leaves"],
                        dx, dy, color_space, method_fn, delta, threshold, alpha)
        total_pruned += n
        if n == 0:
            break

    return jsonify({**_tree_response(session), "pruned": total_pruned})


def main():
    app.run(debug=True, port=5000, host="0.0.0.0")


if __name__ == "__main__":
    main()
