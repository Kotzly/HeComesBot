"""Minimal real-time renderer.

A stripped-down evaluation backend for ``hecomes-video-fast``.  It renders the
same expression trees as :mod:`hecomes.artgen.tree` and, with bilinear
sampling, produces bit-identical output — both backends now compile their
warps through :mod:`hecomes.artgen.resample` and blur through the separable
kernels in :mod:`hecomes.artgen.func_utils`.

What is left that is specific to this backend:

1. **In-place elementwise ops.**  Every buffer in a plan has exactly one
   consumer, so unary and binary elementwise ops write into their input
   instead of allocating.  The general backend cannot assume this: the web UI
   evaluates and re-evaluates subtrees of a shared node graph.
2. **Nearest sampling.**  ``--sampling nearest`` drops bilinear interpolation
   for about a quarter of the cost per warp, which is the lever that makes
   540x960 hold 30 fps on a warp-heavy tree.
3. **A restricted operator set,** so nothing in a plan can be slow by
   surprise, plus the frame-rate budget in the CLI that times and redraws
   trees that would fall behind.

Together those are worth roughly 1.2x over the general backend with bilinear
sampling and 2.1x with nearest, measured at 540x960.  Before the shared
optimisations landed the gap was 2.8x; most of that has since moved into the
general pipeline, where every caller gets it.

Operators that are inherently per-frame (``warp_by``, ``hsv_warp``: the warp
field is itself animated) or asymptotically expensive
(``swap_phase_amplitude``: two FFTs per frame) have no fast path and are
excluded from the op set — see :data:`FAST_FUNCTIONS`.
:func:`restrict_weights` zeroes them out of a personality so they are never
drawn.
"""

import numpy as np
from scipy.spatial.transform import Rotation as R

from hecomes.artgen.func_utils import separable_blur, separable_sharpen
from hecomes.artgen.functions import FUNCTION_REGISTRY
from hecomes.artgen.resample import WARP_MAPS, apply_resampler, compile_resampler

F32 = np.float32


def _op_resample(ctx, a):
    return apply_resampler(ctx, a)


# ── Elementwise ops (in-place: each buffer has exactly one consumer) ──────────


def _op_sin(ctx, a):
    return np.sin(a, out=a)


def _op_cos(ctx, a):
    return np.cos(a, out=a)


def _op_abs(ctx, a):
    return np.abs(a, out=a)


def _op_sigmoid(ctx, a):
    a -= F32(0.5)
    a *= F32(-6.0)
    np.exp(a, out=a)
    a += F32(1.0)
    return np.reciprocal(a, out=a)


def _op_mirrored_sigmoid(ctx, a):
    np.exp(a, out=a)
    a += F32(1.0)
    return np.reciprocal(a, out=a)


def _op_blur(ctx, a):
    return separable_blur(a, output=a)


def _op_sharpen(ctx, a):
    blurred = separable_blur(a)
    a *= F32(2.0)
    a -= blurred
    return a


def _op_color_rotate(ctx, a):
    n, dy, dx, c = a.shape
    if c != 3:
        return a
    return (a.reshape(-1, 3) @ ctx).reshape(n, dy, dx, c)


def _op_add(ctx, a, b):
    a += b
    return a


def _op_subtract(ctx, a, b):
    a -= b
    return a


def _op_multiply(ctx, a, b):
    a *= b
    return a


def _op_saddle(ctx, a, b):
    a *= a
    b *= b
    a -= b
    return a


def _op_safe_divide(ctx, a, b, eps=1e-3):
    with np.errstate(divide="ignore", invalid="ignore"):
        out = a / b
    np.copyto(out, np.sign(b) * F32(1.0 / eps), where=np.isinf(out))
    np.copyto(out, F32(0.0), where=np.isnan(out))
    return out


def _op_safe_modulus(ctx, a, b):
    return np.mod(a, np.where(b == 0, F32(1e-10), b))


def _op_circular_mean(ctx, a, b):
    a1, a2 = a * F32(2 * np.pi), b * F32(2 * np.pi)
    return (np.arctan2(np.sin(a1) + np.sin(a2), np.cos(a1) + np.cos(a2)) / F32(2 * np.pi)) % F32(1.0)


def _op_circular_mean_far(ctx, a, b):
    return (_op_circular_mean(ctx, a, b) + F32(0.5)) % F32(1.0)


def _op_hue_diff(ctx, a, b):
    a -= b
    np.abs(a, out=a)
    a %= F32(1.0)
    return np.minimum(a, F32(1.0) - a, out=a)


def _op_hue_rotate(ctx, a, b):
    a += b
    a %= F32(1.0)
    return a


_LUMA = np.array([0.2126, 0.7152, 0.0722], dtype=F32)


def _op_blend(ctx, a, b, mask):
    np.clip(mask, 0.0, 1.0, out=mask)
    a *= mask
    mask *= F32(-1.0)
    mask += F32(1.0)
    b *= mask
    a += b
    return a


def _op_rgb_compose(ctx, a, b, c):
    out = np.empty_like(a)
    out[..., 0] = (a * _LUMA).sum(axis=-1)
    out[..., 1] = (b * _LUMA).sum(axis=-1)
    out[..., 2] = (c * _LUMA).sum(axis=-1)
    return out


_ELEMENTWISE = {
    "sin": _op_sin,
    "cos": _op_cos,
    "absolute_value": _op_abs,
    "sigmoid": _op_sigmoid,
    "mirrored_sigmoid": _op_mirrored_sigmoid,
    "blur": _op_blur,
    "sharpen": _op_sharpen,
    "add": _op_add,
    "subtract": _op_subtract,
    "multiply": _op_multiply,
    "saddle": _op_saddle,
    "safe_divide": _op_safe_divide,
    "safe_modulus": _op_safe_modulus,
    "circular_mean": _op_circular_mean,
    "circular_mean_far": _op_circular_mean_far,
    "hue_diff": _op_hue_diff,
    "hue_rotate": _op_hue_rotate,
    "blend": _op_blend,
    "rgb_compose": _op_rgb_compose,
}

# Leaves are rendered once at build time by the generic primitives, so every
# arity-0 function is supported as-is.
_LEAF_FUNCTIONS = frozenset(fd.func.__name__ for fd in FUNCTION_REGISTRY if fd.arity == 0)

#: Function names the fast backend can evaluate.  Anything else is dropped from
#: the personality by :func:`restrict_weights`.
FAST_FUNCTIONS = frozenset(
    _LEAF_FUNCTIONS | set(_ELEMENTWISE) | set(WARP_MAPS) | {"color_rotate"}
)


def restrict_weights(weights):
    """Zero out unsupported functions in a personality weight list.

    ``weights`` is a list ordered as :func:`hecomes.config.load_personality_list`
    produces it (sorted by function name).  Returns ``(weights, dropped_names)``
    where ``dropped_names`` lists the unsupported functions that had a nonzero
    weight, so the caller can warn about them.
    """
    names = sorted(fd.func.__name__ for fd in FUNCTION_REGISTRY)
    arities = {fd.func.__name__: fd.arity for fd in FUNCTION_REGISTRY}
    restricted, dropped = [], []
    for name, w in zip(names, weights):
        if name in FAST_FUNCTIONS:
            restricted.append(float(w))
        else:
            restricted.append(0.0)
            if w:
                dropped.append(name)

    # Tree building draws leaves and inner nodes from disjoint slices of this
    # list, so an empty slice would leave it with nothing to sample from.
    leaves = sum(w for n, w in zip(names, restricted) if arities[n] == 0)
    inner = sum(w for n, w in zip(names, restricted) if arities[n] > 0)
    if not leaves or not inner:
        missing = "leaf" if not leaves else "non-leaf"
        raise ValueError(
            f"Personality has no {missing} function the fast backend supports "
            f"(dropped: {', '.join(dropped) or 'none'}). "
            f"Add one of: {', '.join(sorted(FAST_FUNCTIONS))}."
        )
    return restricted, dropped


# ── Plan compilation ──────────────────────────────────────────────────────────


def compile_fast(order, nodes, leaves, dx, dy, bilinear=True):
    """Compile a topologically ordered tree into a flat, pre-computed plan.

    Each entry is ``(fn, ctx, child_indices)``; leaves use ``fn=None`` and carry
    ``ctx=(base_array, delta)``.  Everything in the plan is a plain array, float
    or top-level function, so the plan pickles cleanly and can be handed to
    worker processes under any start method.
    """
    id_to_idx = {nid: i for i, nid in enumerate(order)}
    plan = []
    for nid in order:
        node = nodes[nid]
        name = node.func.func.__name__
        if node.arity == 0:
            plan.append((None, (leaves[nid].astype(F32, copy=False), F32(node.delta)), []))
            continue

        children = [id_to_idx[cid] for cid in node.children]
        if name in WARP_MAPS:
            x_src, y_src = WARP_MAPS[name](dx, dy, **node.params)
            ctx = compile_resampler(x_src, y_src, dx, dy, bilinear)
            plan.append((_op_resample, ctx, children))
        elif name == "color_rotate":
            matrix = R.from_euler("zyx", node.params["angles"]).as_matrix().T.astype(F32)
            plan.append((_op_color_rotate, matrix, children))
        elif name in _ELEMENTWISE:
            plan.append((_ELEMENTWISE[name], None, children))
        else:
            raise ValueError(
                f"'{name}' has no fast implementation. "
                f"Filter the personality through restrict_weights() first."
            )
    return plan


def eval_fast(plan, steps):
    """Evaluate a compiled plan for one chunk of frames.

    ``steps`` is the frame time column, shape ``(n_frames, 1, 1, 1)``.  Returns
    a ``(n_frames, dy, dx, 3)`` float32 array.  Buffers are released as soon as
    their parent consumes them, so peak memory is O(tree depth) chunks.
    """
    buf = [None] * len(plan)
    for i, (fn, ctx, children) in enumerate(plan):
        if fn is None:
            base, delta = ctx
            buf[i] = base + delta * steps
        else:
            args = [buf[j] for j in children]
            for j in children:
                buf[j] = None
            buf[i] = fn(ctx, *args)
    return buf[-1]
