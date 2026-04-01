import uuid
from dataclasses import dataclass, field
from os.path import isfile

import numpy as np

from hecomes.artgen.functions import FUNCTION_REGISTRY, REGISTRY_BY_NAME, FunctionDef


def random_delta(alpha=5e-3):
    return np.random.choice([1, -1]) * alpha


# ── Node ──────────────────────────────────────────────────────────────────────


@dataclass
class Node:
    func: FunctionDef
    params: dict = field(default_factory=dict)
    children: list[str] = field(default_factory=list)  # child node IDs
    delta: float | None = None
    id: str = field(default_factory=lambda: str(uuid.uuid4())[:8])

    @property
    def arity(self) -> int:
        return self.func.arity

    def to_dict(self) -> dict:
        """Serialize this node's own data (no recursion — children are IDs)."""
        d = {
            "func": self.func.func.__name__,
            "arity": self.arity,
            "params": self.params,
            "children": self.children,
        }
        if self.delta is not None:
            d["delta"] = self.delta
        return d

    @classmethod
    def from_dict(cls, node_id: str, d: dict) -> "Node":
        fd = REGISTRY_BY_NAME[d["func"]]
        return cls(
            func=fd,
            params=d.get("params", {}),
            children=list(d.get("children", [])),
            delta=d.get("delta"),
            id=node_id,
        )


def nodes_to_dict(nodes: dict) -> dict:
    return {nid: node.to_dict() for nid, node in nodes.items()}


def nodes_from_dict(d: dict) -> dict:
    return {nid: Node.from_dict(nid, entry) for nid, entry in d.items()}


# ── Tree building ──────────────────────────────────────────────────────────────


def get_random_function(depth=0, min_depth=5, max_depth=15, p=None) -> FunctionDef:
    sorted_registry = sorted(FUNCTION_REGISTRY, key=lambda fd: fd.func.__name__)
    if p is None:
        p = np.ones(len(sorted_registry))

    candidates, weights = [], []
    for fd, w in zip(sorted_registry, p):
        if (fd.arity > 0 and depth < max_depth) or (fd.arity == 0 and depth >= min_depth):
            candidates.append(fd)
            weights.append(w)

    weights = np.array(weights, dtype=float)
    weights /= weights.sum()
    return candidates[np.random.choice(len(candidates), p=weights)]


def build_node(depth, min_depth, max_depth, dx, dy, weights, alpha, nodes, leaves) -> str:
    """Build a subtree, populating `nodes` and `leaves`. Returns the root node ID."""
    fd = get_random_function(depth, p=weights, min_depth=min_depth, max_depth=max_depth)
    params = fd.generate() if fd.generate else {}
    if fd.arity == 0:
        node = Node(func=fd, params=params, delta=float(random_delta(alpha)))
        leaves[node.id] = fd.func(dx=dx, dy=dy, **params).astype(np.float32)
    else:
        child_ids = [
            build_node(depth + 1, min_depth, max_depth, dx, dy, weights, alpha, nodes, leaves)
            for _ in range(fd.arity)
        ]
        node = Node(func=fd, params=params, children=child_ids)
    nodes[node.id] = node
    return node.id


# ── Tree evaluation ────────────────────────────────────────────────────────────


def eval_node(node_id: str, nodes: dict, leaves: dict, steps) -> np.ndarray:
    node = nodes[node_id]
    if node.arity == 0:
        return leaves[node_id] + node.delta * steps
    args = [eval_node(cid, nodes, leaves, steps) for cid in node.children]
    return node.func.func(*args, **node.params)


# ── Tree traversal ─────────────────────────────────────────────────────────────


def collect_leaf_ids(node_id: str, nodes: dict) -> list[str]:
    node = nodes[node_id]
    if node.arity == 0:
        return [node_id]
    ids = []
    for cid in node.children:
        ids.extend(collect_leaf_ids(cid, nodes))
    return ids


def find_node(nodes: dict, node_id: str):
    return nodes.get(node_id)


def find_parent(nodes: dict, node_id: str):
    """Return (parent_id, child_index) or None if node_id is the root."""
    for nid, node in nodes.items():
        if node_id in node.children:
            return nid, node.children.index(node_id)
    return None


def node_depth(nodes: dict, root_id: str, target_id: str):
    queue = [(root_id, 0)]
    while queue:
        nid, d = queue.pop(0)
        if nid == target_id:
            return d
        for cid in nodes[nid].children:
            queue.append((cid, d + 1))
    return None


# ── Logging ────────────────────────────────────────────────────────────────────


def log_tree_to_file(root_id: str, nodes: dict, depth: int = 0, log_filepath="tree.txt"):
    if log_filepath is None:
        return
    node = nodes[root_id]
    mode = "a" if isfile(log_filepath) else "w"
    with open(log_filepath, mode) as log_file:
        log_file.write("|\t" * depth + node.func.func.__name__ + "\n")
    for cid in node.children:
        log_tree_to_file(cid, nodes, depth + 1, log_filepath)
