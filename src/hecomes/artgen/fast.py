"""Minimal real-time renderer.

A stripped-down evaluation backend for ``hecomes-video-fast``.  It renders the
same expression trees as :mod:`hecomes.artgen.tree`, but trades generality for
speed so that generation keeps up with playback (>= 30 fps at 540x960).

Three ideas carry the whole speedup:

1. **Static warps are pre-compiled.**  ``swirl``/``ripple``/``pinch``/
   ``polar_warp``/``kaleidoscope`` all resolve to a backward map that depends
   only on the node parameters and the frame size — never on time.  The sample
   coordinates (and bilinear weights) are computed once at compile time and the
   per-frame work collapses to a gather.  The general backend instead rebuilds
   the coordinate grid and calls ``scipy.ndimage.map_coordinates`` once per
   channel per frame: 30 calls per 10-frame chunk against 1 gather here.
2. **Separable convolution.**  ``blur``/``sharpen`` use a 5x5 binomial kernel,
   which factors into two 1D passes (10 multiply-adds per pixel instead of 25),
   applied to the whole chunk at once rather than per frame per channel.
3. **In-place elementwise ops.**  Every buffer in the plan has exactly one
   consumer, so unary and binary elementwise ops write into their input instead
   of allocating. Roughly halves the memory traffic of ``sin``/``cos``/``abs``.

Everything runs in float32 end to end.

Operators that are inherently per-frame (``warp_by``, ``hsv_warp``: the warp
field is itself animated) or asymptotically expensive (``swap_phase_amplitude``:
two FFTs per frame) have no fast path and are excluded from the op set — see
:data:`FAST_FUNCTIONS`.  :func:`restrict_weights` zeroes them out of a
personality so they are never drawn.
"""

import numpy as np
from scipy.ndimage import convolve1d
from scipy.spatial.transform import Rotation as R

from hecomes.artgen.functions import FUNCTION_REGISTRY

F32 = np.float32

# 5-tap binomial kernel — the separable factor of the 5x5 kernel in func_utils.
_BINOMIAL_5 = np.array([1, 4, 6, 4, 1], dtype=F32) / 16.0


# ── Coordinate grids ──────────────────────────────────────────────────────────


def _mesh(dx, dy):
    """float32 [-1, 1] coordinate grid, broadcast (no per-pixel materialisation)."""
    xs = np.linspace(-1.0, 1.0, dx, dtype=F32)[None, :]
    ys = np.linspace(-1.0, 1.0, dy, dtype=F32)[:, None]
    return np.broadcast_to(xs, (dy, dx)), np.broadcast_to(ys, (dy, dx))


def _compile_resampler(x_src, y_src, dx, dy, bilinear):
    """Turn a backward map in [-1, 1] space into a pre-computed gather context.

    Returns ``(indices, weights)``: a list of flat pixel indices and matching
    weight columns.  ``nearest`` yields one index and no weights; ``bilinear``
    yields the four corners and their areas.  Out-of-bounds samples are clamped
    to the edge, matching ``map_coordinates(mode="nearest")``.
    """
    col = np.clip((x_src + 1.0) * 0.5 * (dx - 1), 0, dx - 1)
    row = np.clip((y_src + 1.0) * 0.5 * (dy - 1), 0, dy - 1)

    if not bilinear:
        idx = (np.rint(row).astype(np.intp) * dx + np.rint(col).astype(np.intp)).ravel()
        return [idx], None

    c0 = np.floor(col).astype(np.intp)
    r0 = np.floor(row).astype(np.intp)
    c1 = np.minimum(c0 + 1, dx - 1)
    r1 = np.minimum(r0 + 1, dy - 1)
    fc = (col - c0).astype(F32)
    fr = (row - r0).astype(F32)
    indices = [
        (r0 * dx + c0).ravel(),
        (r0 * dx + c1).ravel(),
        (r1 * dx + c0).ravel(),
        (r1 * dx + c1).ravel(),
    ]
    weights = [
        ((1 - fr) * (1 - fc)).ravel()[:, None],
        ((1 - fr) * fc).ravel()[:, None],
        (fr * (1 - fc)).ravel()[:, None],
        (fr * fc).ravel()[:, None],
    ]
    return indices, weights


def _op_resample(ctx, a):
    indices, weights = ctx
    n, dy, dx, c = a.shape
    flat = a.reshape(n, dy * dx, c)
    if weights is None:
        return np.take(flat, indices[0], axis=1).reshape(n, dy, dx, c)
    out = np.take(flat, indices[0], axis=1)
    out *= weights[0]
    for i in (1, 2, 3):
        part = np.take(flat, indices[i], axis=1)
        part *= weights[i]
        out += part
    return out.reshape(n, dy, dx, c)


# ── Backward maps (compile time only) ─────────────────────────────────────────


def _map_swirl(dx, dy, cx=0.0, cy=0.0, strength=1.0, power=-2.0):
    xs, ys = _mesh(dx, dy)
    rx, ry = xs - cx, ys - cy
    r = np.sqrt(rx * rx + ry * ry) + 1e-6
    theta = np.arctan2(ry, rx) - strength * r**power
    return cx + r * np.cos(theta), cy + r * np.sin(theta)


def _map_ripple(dx, dy, ax=0.1, ay=0.1, kx=4.0, ky=4.0, phase_x=0.0, phase_y=0.0):
    xs, ys = _mesh(dx, dy)
    return xs + ax * np.sin(kx * ys + phase_x), ys + ay * np.sin(ky * xs + phase_y)


def _map_pinch(dx, dy, cx=0.0, cy=0.0, strength=0.5):
    xs, ys = _mesh(dx, dy)
    rx, ry = xs - cx, ys - cy
    r = np.sqrt(rx * rx + ry * ry) + 1e-6
    scale = np.power(r, 1.0 + strength) / r
    return cx + rx * scale, cy + ry * scale


def _map_polar_warp(dx, dy, cx=0.0, cy=0.0):
    xs, ys = _mesh(dx, dy)
    rx, ry = xs - cx, ys - cy
    r = np.sqrt(rx * rx + ry * ry)
    theta = (np.arctan2(ry, rx) + 2 * np.pi) % (2 * np.pi)
    # angle indexes the source's horizontal axis, radius its vertical axis
    col = theta / (2 * np.pi) * 2.0 - 1.0
    row = r / np.sqrt(2.0) * 2.0 - 1.0
    return col, row


def _map_kaleidoscope(dx, dy, n=6, phase=0.0):
    """Fold the plane into ``n`` mirrored wedges.

    Expressed as a plain backward map: the general backend reaches the same
    picture through a ``NearestNDInterpolator`` over the (angle, radius) point
    cloud, which builds a KD-tree per frame. Sampling is nearest-neighbour
    there and follows ``--sampling`` here, so edges may differ by a pixel.
    """
    xs, ys = _mesh(dx, dy)
    phi = 2 * np.pi / n
    angle = np.arctan2(-ys, xs) % (2 * np.pi)
    r = np.sqrt(xs * xs + ys * ys)
    folded = angle % phi + phase
    return r * np.cos(folded), -r * np.sin(folded)


_WARP_MAPS = {
    "swirl": _map_swirl,
    "ripple": _map_ripple,
    "pinch": _map_pinch,
    "polar_warp": _map_polar_warp,
    "kaleidoscope": _map_kaleidoscope,
}


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
    out = convolve1d(a, _BINOMIAL_5, axis=1, mode="reflect", output=a)
    return convolve1d(out, _BINOMIAL_5, axis=2, mode="reflect", output=out)


def _op_sharpen(ctx, a):
    """Unsharp mask: 2*a - blur(a), the separable form of the 5x5 sharpen kernel."""
    blurred = convolve1d(a, _BINOMIAL_5, axis=1, mode="reflect")
    convolve1d(blurred, _BINOMIAL_5, axis=2, mode="reflect", output=blurred)
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
    _LEAF_FUNCTIONS | set(_ELEMENTWISE) | set(_WARP_MAPS) | {"color_rotate"}
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
        if name in _WARP_MAPS:
            x_src, y_src = _WARP_MAPS[name](dx, dy, **node.params)
            ctx = _compile_resampler(x_src, y_src, dx, dy, bilinear)
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
