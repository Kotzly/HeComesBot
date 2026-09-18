"""Backward-map resampling, shared by every pixel-warp operator.

All warps in this project are backward maps: for each output pixel at
``(x, y)`` they compute a source coordinate ``(x_src, y_src)`` in ``[-1, 1]``
space and sample the input there.  The map depends only on the operator's
parameters and the frame size — never on the pixel data and never on time —
so it is worth separating from the sampling:

- :func:`compile_resampler` turns a map into a context of flat indices and
  fractions, once.
- :func:`apply_resampler` applies that context to a whole ``(n, dy, dx, c)``
  batch in one pass.

:mod:`hecomes.artgen.primitives` compiles per call, which already collapses the
old one-``map_coordinates``-call-per-channel-per-frame loop into a single
gather.  :mod:`hecomes.artgen.fast` compiles once per video and reuses the
context across every chunk.

Contexts are deliberately compact.  Storing the four corner indices and their
four weights costs eight full-resolution arrays — 398 MB per warp node at 4K.
Clamping the top-left corner to ``dx - 2`` / ``dy - 2`` instead makes the other
three corners ``base + 1``, ``base + dx`` and ``base + dx + 1``, so one int32
index array and two float32 fractions suffice: 100 MB at 4K, for 7% more time
in the gather.  The clamp is exact, not an approximation — at ``col = dx - 1``
it yields ``c0 = dx - 2`` with ``fc = 1``, which samples ``dx - 1`` at full
weight.
"""

import numpy as np

from hecomes.artgen.func_utils import linear_mesh

F32 = np.float32


# ── Backward maps ─────────────────────────────────────────────────────────────


def swirl_map(dx, dy, cx=0.0, cy=0.0, strength=1.0, power=-2.0):
    """Rotate around ``(cx, cy)`` by ``strength * r**power``."""
    xs, ys = linear_mesh(dx=dx, dy=dy)
    rx, ry = xs - cx, ys - cy
    r = np.sqrt(rx * rx + ry * ry) + 1e-6
    theta = np.arctan2(ry, rx) - strength * r**power
    return cx + r * np.cos(theta), cy + r * np.sin(theta)


def ripple_map(dx, dy, ax=0.1, ay=0.1, kx=4.0, ky=4.0, phase_x=0.0, phase_y=0.0):
    """Sinusoidal displacement along each axis."""
    xs, ys = linear_mesh(dx=dx, dy=dy)
    return xs + ax * np.sin(kx * ys + phase_x), ys + ay * np.sin(ky * xs + phase_y)


def pinch_map(dx, dy, cx=0.0, cy=0.0, strength=0.5):
    """Radial lens distortion: ``r_src = r ** (1 + strength)``."""
    xs, ys = linear_mesh(dx=dx, dy=dy)
    rx, ry = xs - cx, ys - cy
    r = np.sqrt(rx * rx + ry * ry) + 1e-6
    scale = np.power(r, 1.0 + strength) / r
    return cx + rx * scale, cy + ry * scale


def polar_warp_map(dx, dy, cx=0.0, cy=0.0):
    """Cartesian-to-polar remap: angle indexes columns, radius indexes rows."""
    xs, ys = linear_mesh(dx=dx, dy=dy)
    rx, ry = xs - cx, ys - cy
    r = np.sqrt(rx * rx + ry * ry)
    theta = (np.arctan2(ry, rx) + 2 * np.pi) % (2 * np.pi)
    # r_max = sqrt(2) covers the full [-1, 1]^2 diagonal
    return theta / (2 * np.pi) * 2.0 - 1.0, r / np.sqrt(2.0) * 2.0 - 1.0


def kaleidoscope_map(dx, dy, n=6, phase=0.0):
    """Fold the plane into ``n`` wedges around the origin."""
    xs, ys = linear_mesh(dx=dx, dy=dy)
    phi = 2 * np.pi / n
    angle = np.arctan2(-ys, xs) % (2 * np.pi)
    r = np.sqrt(xs * xs + ys * ys)
    folded = angle % phi + phase
    return r * np.cos(folded), -r * np.sin(folded)


WARP_MAPS = {
    "swirl": swirl_map,
    "ripple": ripple_map,
    "pinch": pinch_map,
    "polar_warp": polar_warp_map,
    "kaleidoscope": kaleidoscope_map,
}


# ── Compile / apply ───────────────────────────────────────────────────────────


def compile_resampler(x_src, y_src, dx, dy, bilinear=True):
    """Turn a backward map in ``[-1, 1]`` space into a reusable sampling context.

    Out-of-bounds coordinates are clamped to the edge, matching
    ``map_coordinates(..., mode="nearest")``.  Returns an opaque tuple for
    :func:`apply_resampler`; the layout differs between the two modes.
    """
    col = np.clip((x_src + 1.0) * 0.5 * (dx - 1), 0, dx - 1)
    row = np.clip((y_src + 1.0) * 0.5 * (dy - 1), 0, dy - 1)

    if not bilinear:
        base = (np.rint(row) * dx + np.rint(col)).astype(np.int32).ravel()
        return (False, base, None, None, dx, dy)

    # Clamping one short of the edge keeps the remaining three corners at
    # base + 1 / base + dx / base + dx + 1 without a separate index array.
    c0 = np.minimum(np.floor(col), dx - 2)
    r0 = np.minimum(np.floor(row), dy - 2)
    fc = (col - c0).astype(F32).ravel()[:, None]
    fr = (row - r0).astype(F32).ravel()[:, None]
    base = (r0 * dx + c0).astype(np.int32).ravel()
    return (True, base, fr, fc, dx, dy)


def apply_resampler(ctx, batch):
    """Sample ``batch`` (n, dy, dx, c) through a compiled context.

    Corners are accumulated one at a time so peak memory stays at two
    full-size buffers rather than four — the difference between 232 MB and
    398 MB for a single 4K frame.
    """
    bilinear, base, fr, fc, dx, dy = ctx
    n, _, _, channels = batch.shape
    flat = batch.reshape(n, dy * dx, channels)

    if not bilinear:
        return np.take(flat, base, axis=1).reshape(n, dy, dx, channels)

    one_minus_fr = 1.0 - fr
    one_minus_fc = 1.0 - fc
    out = np.take(flat, base, axis=1)
    out *= one_minus_fr * one_minus_fc
    for offset, weight in (
        (1, one_minus_fr * fc),
        (dx, fr * one_minus_fc),
        (dx + 1, fr * fc),
    ):
        corner = np.take(flat, base + offset, axis=1)
        corner *= weight
        out += corner
    return out.reshape(n, dy, dx, channels)


def warp(name, batch, bilinear=True, **params):
    """Compile ``name``'s backward map for ``batch``'s size and apply it."""
    _, dy, dx, _ = batch.shape
    x_src, y_src = WARP_MAPS[name](dx, dy, **params)
    return apply_resampler(compile_resampler(x_src, y_src, dx, dy, bilinear), batch)


def apply_compiled(batch, ctx=None):
    """Adapter that takes the context as a keyword argument.

    The compiled plans in :mod:`hecomes.artgen.tree` and
    :mod:`hecomes.artgen.tree_paths` call inner nodes as
    ``func(*children, **params)``.  Passing a pre-compiled context as one of
    those params lets a warp node be swapped in without changing either plan
    format, and the context is plain arrays, so the plan still pickles.
    """
    return apply_resampler(ctx, batch)


def compile_warp_params(func_name, params, dx, dy, bilinear=True):
    """Pre-compile a warp node's backward map, or return ``None``.

    Returns the ``params`` dict to store in the plan (``{"ctx": ...}``) when
    ``func_name`` is a static warp, and ``None`` when it is not one — including
    when the stored params do not match the map's signature, which leaves the
    caller on the general path rather than failing to build the video.
    """
    if func_name not in WARP_MAPS:
        return None
    try:
        x_src, y_src = WARP_MAPS[func_name](dx, dy, **params)
    except TypeError:
        return None
    return {"ctx": compile_resampler(x_src, y_src, dx, dy, bilinear)}
