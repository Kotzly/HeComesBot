"""Path-based tree building and evaluation (new animation backend).

The old backend (``tree.py``) is completely untouched.  This module provides
parallel functions that use :class:`~hecomes.artgen.paths.Path` objects instead
of pre-rendered leaf arrays.

Compiled plan format
--------------------
:func:`compile_plan_paths` returns a list of tuples, one per node in
topological order (leaves first, root last):

.. code-block:: text

    Leaf:  (True,  primitive_func, dx, dy, paths,  default_params, None,   [])
    Inner: (False, func,           0,  0,  [],     {},             params, child_idxs)

Fields (by index):

0. ``is_leaf``        bool
1. ``func``           callable
2. ``dx``             int  (leaf only)
3. ``dy``             int  (leaf only)
4. ``paths``          list[Path]  (leaf: path objects; inner: empty)
5. ``default_params`` dict  (leaf: build-time values for non-animated params)
6. ``params``         dict  (inner: fixed call params; leaf: None)
7. ``child_idxs``     list[int]
"""

from __future__ import annotations

import json
import uuid

import numpy as np

try:
    import cupy as cp
except ImportError:
    cp = None

from hecomes.artgen.functions import FunctionDef
from hecomes.artgen.paths import (
    AngularVelocity,
    CircularOrbit,
    HuePath,
    ODEPath,
    Oscillate,
    Path,
    path_from_dict,
)
from hecomes.artgen.tree import Node, get_random_function, linearize


# ── Personality helpers ───────────────────────────────────────────────────────

_DEFAULT_PATHS_CONFIG: dict = {
    "animation_probability": 0.5,
    "path_weights": {
        "CircularOrbit": 3.0,
        "Oscillate": 2.0,
        "AngularVelocity": 2.0,
        "HuePath": 1.5,
        "LinearDrift": 1.0,
    },
    "omega_range": [0.5, 4.0],
    "amplitude_range": [0.05, 0.4],
}


def load_paths_config(personality_path) -> dict:
    """Read the ``"paths"`` section from a personality JSON file.

    Falls back to :data:`_DEFAULT_PATHS_CONFIG` if the file does not exist,
    cannot be parsed, or does not contain a ``"paths"`` key.
    """
    if personality_path is None:
        return _DEFAULT_PATHS_CONFIG
    try:
        with open(personality_path) as f:
            data = json.load(f)
        return data.get("paths", _DEFAULT_PATHS_CONFIG)
    except (FileNotFoundError, json.JSONDecodeError):
        return _DEFAULT_PATHS_CONFIG


def _effective_config(fd: FunctionDef, paths_config: dict) -> dict:
    """Merge top-level paths config with per-primitive overrides.

    Per-primitive keys override the top-level values; any key absent from the
    per-primitive section falls back to the top-level default.
    """
    per_prim = paths_config.get("primitives", {}).get(fd.func.__name__, {})
    merged = dict(paths_config)
    merged.update(per_prim)
    return merged


# ── Path generation ───────────────────────────────────────────────────────────

def _rgb_to_hsv(rgb: list) -> tuple[float, float, float]:
    """Simple RGB → HSV conversion for a single [r, g, b] triple."""
    r, g, b = float(rgb[0]), float(rgb[1]), float(rgb[2])
    cmax = max(r, g, b)
    cmin = min(r, g, b)
    delta = cmax - cmin
    if delta < 1e-6:
        h = 0.0
    elif cmax == r:
        h = ((g - b) / delta) % 6.0 / 6.0
    elif cmax == g:
        h = ((b - r) / delta + 2.0) / 6.0
    else:
        h = ((r - g) / delta + 4.0) / 6.0
    s = float(delta / cmax) if cmax > 1e-6 else 0.0
    v = float(cmax)
    return h, s, v


def sample_paths_for_leaf(
    fd: FunctionDef,
    params: dict,
    paths_config: dict,
    duration: float,
) -> list[Path]:
    """Randomly generate :class:`~hecomes.artgen.paths.Path` objects for a leaf.

    Uses the per-primitive override from ``paths_config["primitives"]`` if
    present, falling back to the top-level config for any missing key.

    Returns a list of Path objects.  Params not covered by any path retain their
    build-time ``params`` values (stored as ``default_params`` in the plan).
    """
    cfg = _effective_config(fd, paths_config)
    anim_prob: float = cfg.get("animation_probability", 0.5)
    pw: dict = cfg.get("path_weights", {})
    omega_lo, omega_hi = cfg.get("omega_range", [0.5, 4.0])
    amp_lo, amp_hi = cfg.get("amplitude_range", [0.05, 0.4])

    def _omega() -> float:
        return float(np.random.uniform(omega_lo, omega_hi))

    def _amp() -> float:
        return float(np.random.uniform(amp_lo, amp_hi))

    def _phase() -> float:
        return float(np.random.uniform(0.0, 2.0 * np.pi))

    def _animate() -> bool:
        return np.random.rand() < anim_prob

    animatable = [p for p in fd.params if p.get("animatable", False)]
    float_params = {p["name"]: p for p in animatable if p["type"] == "float"}
    color_params = [p for p in animatable if p["type"] == "color"]

    paths: list[Path] = []
    used: set[str] = set()

    # ── Position: CircularOrbit for (cx, cy) ──────────────────────────────────
    has_cx = "cx" in float_params
    has_cy = "cy" in float_params
    w_orbit = pw.get("CircularOrbit", 0.0)
    if has_cx and has_cy and w_orbit > 0 and _animate():
        w_osc = pw.get("Oscillate", 0.0) + pw.get("LinearDrift", 0.0)
        total = w_orbit + w_osc
        if total > 0 and np.random.rand() < w_orbit / total:
            paths.append(CircularOrbit(
                cx0=float(params.get("cx", 0.0)),
                cy0=float(params.get("cy", 0.0)),
                r=_amp() * 0.5,
                omega=_omega(),
                phase=_phase(),
            ))
            used |= {"cx", "cy"}

    # ── Angle: AngularVelocity ─────────────────────────────────────────────────
    w_av = pw.get("AngularVelocity", 0.0)
    if "angle" in float_params and "angle" not in used and w_av > 0 and _animate():
        paths.append(AngularVelocity(
            angle0=float(params.get("angle", 0.0)),
            omega=_omega(),
        ))
        used.add("angle")

    # ── Remaining float params: Oscillate ────────────────────────────────────
    w_osc = pw.get("Oscillate", 0.0)
    if w_osc > 0:
        for name, pmeta in float_params.items():
            if name in used:
                continue
            if not _animate():
                continue
            base = float(params.get(name, 0.0))
            pmin = pmeta.get("min", base - 1.0)
            pmax = pmeta.get("max", base + 1.0)
            max_amp = min(_amp(), (pmax - pmin) * 0.25)
            paths.append(Oscillate(
                param=name,
                base=base,
                amplitude=max_amp,
                omega=_omega(),
                phase=_phase(),
            ))
            used.add(name)

    # ── Color: HuePath ────────────────────────────────────────────────────────
    w_hue = pw.get("HuePath", 0.0)
    if w_hue > 0:
        for pmeta in color_params:
            if pmeta["name"] in used:
                continue
            if not _animate():
                continue
            h0, s0, v0 = _rgb_to_hsv(params.get(pmeta["name"], [0.5, 0.5, 0.5]))
            paths.append(HuePath(
                h0=h0, s0=s0, v0=v0,
                h_speed=float(np.random.uniform(0.3, 1.5)),
                s_A=_amp() * 0.3,
                s_omega=_omega(),
                v_A=_amp() * 0.2,
                v_omega=_omega() * 1.3,
                period=duration,
            ))
            used.add(pmeta["name"])

    return paths


# ── Tree building ─────────────────────────────────────────────────────────────


def build_node_paths(
    depth: int,
    min_depth: int,
    max_depth: int,
    dx: int,
    dy: int,
    weights,
    nodes: dict,
    paths_per_leaf: dict,
    params_per_leaf: dict,
    paths_config: dict,
    duration: float,
) -> str:
    """Build a tree node for the path-based backend.

    Like :func:`~hecomes.artgen.tree.build_node` but instead of pre-rendering
    leaf arrays, stores :class:`~hecomes.artgen.paths.Path` objects in
    ``paths_per_leaf`` and build-time param values in ``params_per_leaf``.

    Returns the root node ID.
    """
    fd = get_random_function(depth, p=weights, min_depth=min_depth, max_depth=max_depth)
    params = fd.generate() if fd.generate else {}
    node_id = str(uuid.uuid4())[:8]

    if fd.arity == 0:
        leaf_paths = sample_paths_for_leaf(fd, params, paths_config, duration)
        paths_per_leaf[node_id] = leaf_paths
        params_per_leaf[node_id] = params
        node = Node(func=fd, params=params, delta=None, id=node_id)
    else:
        child_ids = [
            build_node_paths(
                depth + 1, min_depth, max_depth, dx, dy, weights,
                nodes, paths_per_leaf, params_per_leaf, paths_config, duration,
            )
            for _ in range(fd.arity)
        ]
        node = Node(func=fd, params=params, children=child_ids, id=node_id)

    nodes[node_id] = node
    return node_id


# ── Plan compilation ──────────────────────────────────────────────────────────


def compile_plan_paths(
    order: list,
    nodes: dict,
    paths_per_leaf: dict,
    params_per_leaf: dict,
    dx: int,
    dy: int,
) -> list:
    """Compile a path-based evaluation plan.

    See module docstring for the tuple layout of each entry.
    """
    id_to_idx = {nid: i for i, nid in enumerate(order)}
    plan = []
    for nid in order:
        node = nodes[nid]
        if node.arity == 0:
            plan.append((
                True,
                node.func.func,
                dx,
                dy,
                paths_per_leaf.get(nid, []),
                params_per_leaf.get(nid, {}),
                None,
                [],
            ))
        else:
            child_idxs = [id_to_idx[cid] for cid in node.children]
            plan.append((
                False,
                node.func.func,
                0,
                0,
                [],
                {},
                node.params,
                child_idxs,
            ))
    return plan


# ── ODE pre-integration ───────────────────────────────────────────────────────


def integrate_ode_paths(
    plan: list,
    n_frames: int,
    fps: float,
    solver: str = "rk4",
) -> None:
    """Pre-integrate all :class:`~hecomes.artgen.paths.ODEPath` objects in-place.

    Must be called once after :func:`compile_plan_paths` and before the
    multiprocessing pool is created.  Non-ODE paths are unaffected.

    Parameters
    ----------
    plan:
        Compiled path plan from :func:`compile_plan_paths`.
    n_frames:
        Total number of frames in the video.
    fps:
        Frames per second (determines integration step dt = 1/fps).
    solver:
        ``"euler"`` or ``"rk4"`` (default).
    """
    dt = 1.0 / fps
    for entry in plan:
        is_leaf = entry[0]
        if not is_leaf:
            continue
        for path in entry[4]:  # entry[4] = paths list
            if isinstance(path, ODEPath):
                path.precompute(n_frames, dt, solver)


# ── Evaluation ────────────────────────────────────────────────────────────────


def eval_plan_paths(
    plan: list,
    t_chunk: np.ndarray,
    use_gpu: bool = False,
) -> np.ndarray:
    """Evaluate a compiled path plan for a chunk of time steps.

    Parameters
    ----------
    plan:
        Output of :func:`compile_plan_paths` (with ODEPaths pre-integrated).
    t_chunk:
        Time values in seconds, shape ``(chunk_size, 1, 1, 1)``.
    use_gpu:
        If ``True`` and CuPy is available, inner-node ops run on GPU.  Leaf
        primitives always run on CPU (NumPy); the resulting arrays are moved to
        GPU before inner-node evaluation.

    Returns
    -------
    np.ndarray
        Shape ``(chunk_size, dy, dx, 3)``.
    """
    chunk_size = t_chunk.shape[0]
    buf: list = [None] * len(plan)

    for i, (is_leaf, func, dx, dy, paths, default_params, inner_params, children) in enumerate(plan):
        if is_leaf:
            frames = []
            for k in range(chunk_size):
                t_k = float(t_chunk[k, 0, 0, 0])
                call_params = dict(default_params)
                for path in paths:
                    call_params.update(path(t_k))
                frames.append(func(dx=dx, dy=dy, **call_params))
            stacked = np.stack(frames, axis=0)  # (chunk_size, dy, dx, 3)
            buf[i] = cp.asarray(stacked) if (use_gpu and cp is not None) else stacked
        else:
            args = [buf[j] for j in children]
            for j in children:
                buf[j] = None
            buf[i] = func(*args, **inner_params)

    result = buf[-1]
    if use_gpu and cp is not None:
        return cp.asnumpy(result)
    return result
