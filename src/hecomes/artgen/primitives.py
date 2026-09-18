import numpy as np
from scipy.ndimage import map_coordinates
from scipy.spatial.transform import Rotation as R

try:
    import cupy as cp
except ImportError:
    cp = None

from hecomes.artgen.func_utils import (
    get_radius,
    hsv_to_rgb,
    is_valid_shape,
    linear_mesh,
    random_point,
    random_radius,
    rgb_to_hsv,
    separable_blur,
    separable_sharpen,
)
from hecomes.artgen.resample import warp

# ── Leaf functions (arity 0) ──────────────────────────────────────────────────


def rand_color(dx=None, dy=None, color=None):
    if color is None:
        color = np.random.rand(3).astype(np.float32)
    else:
        color = np.asarray(color, dtype=np.float32)
    return np.broadcast_to(color.reshape(1, 1, 3), (dy, dx, 3)).copy()


def _gen_rand_color():
    return {"color": np.random.rand(3).tolist()}


def _rotated_gradient(dx, dy, angle=None, color=None, px=None, py=None):
    if angle is None:
        angle = np.random.rand() * 2 * np.pi
    if color is None:
        color = np.random.rand(3).astype(np.float32)
    else:
        color = np.asarray(color, dtype=np.float32)
    if px is None:
        px = float(np.random.uniform(-1.0, 1.0))
    if py is None:
        py = float(np.random.uniform(-1.0, 1.0))
    x, y = linear_mesh(dx=dx, dy=dy)
    grad = np.cos(angle) * (x - px) + np.sin(angle) * (y - py)
    return (grad[:, :, np.newaxis] * color.reshape(1, 1, 3)).astype(np.float32)


def x_var(dx=None, dy=None, angle=None, color=None, px=None, py=None):
    return _rotated_gradient(dx, dy, angle=angle, color=color, px=px, py=py)


def y_var(dx=None, dy=None, angle=None, color=None, px=None, py=None):
    return _rotated_gradient(dx, dy, angle=angle, color=color, px=px, py=py)


def _gen_gradient():
    return {
        "angle": float(np.random.rand() * 2 * np.pi),
        "color": np.random.rand(3).tolist(),
        "px": float(np.random.uniform(-1.0, 1.0)),
        "py": float(np.random.uniform(-1.0, 1.0)),
    }


def _cone_gradient(dx, dy, cx, cy, rx, ry):
    x, y = linear_mesh(dx=dx, dy=dy)
    return (
        np.sqrt(((x - cx) / rx) ** 2 + ((y - cy) / ry) ** 2)
        .reshape(dy, dx, 1)
        .astype(np.float32)
    )


def cone(dx=None, dy=None, cx=None, cy=None, rx=None, ry=None, color=None):
    if cx is None or cy is None:
        cx, cy = random_point()
    if rx is None or ry is None:
        rx, ry = random_radius()
    if color is None:
        color = np.random.rand(3).astype(np.float32)
    else:
        color = np.asarray(color, dtype=np.float32)
    gradient = _cone_gradient(dx, dy, cx, cy, rx, ry)
    return (np.broadcast_to(gradient, (dy, dx, 3)) * color.reshape(1, 1, 3)).copy()


def _gen_cone():
    rx, ry = random_radius()
    cx = float(np.random.uniform(-(1 + rx), 1 + rx))
    cy = float(np.random.uniform(-(1 + ry), 1 + ry))
    return {
        "cx": cx, "cy": cy,
        "rx": float(rx), "ry": float(ry),
        "color": np.random.rand(3).tolist(),
    }


def circle(dx=None, dy=None, cx=None, cy=None, rx=None, ry=None, color=None):
    if cx is None or cy is None:
        cx, cy = random_point()
    if rx is None or ry is None:
        rx, ry = random_radius()
    base = _cone_gradient(dx, dy, cx, cy, rx, ry)
    if color is None:
        color = np.random.rand(3).astype(np.float32)
    else:
        color = np.asarray(color, dtype=np.float32)
    circ = np.ones((dy, dx, 3), dtype=np.float32) * color.reshape(1, 1, 3)
    circ[base[:, :, 0] > 1] = 0
    return circ


def sphere(dx=None, dy=None, cx=None, cy=None, rx=None, ry=None, color=None):
    if cx is None or cy is None:
        cx, cy = random_point()
    if rx is None or ry is None:
        rx, ry = random_radius()
    if color is None:
        color = np.random.rand(3).astype(np.float32)
    else:
        color = np.asarray(color, dtype=np.float32)
    x, y = linear_mesh(dx=dx, dy=dy)
    r_sq = ((x - cx) / rx) ** 2 + ((y - cy) / ry) ** 2
    z = np.sqrt(np.maximum(0.0, 1.0 - r_sq)).reshape(dy, dx, 1).astype(np.float32)
    return (np.broadcast_to(z, (dy, dx, 3)) * color.reshape(1, 1, 3)).copy()


def _gen_sphere():
    rx, ry = random_radius()
    cx = float(np.random.uniform(-(1 + rx), 1 + rx))
    cy = float(np.random.uniform(-(1 + ry), 1 + ry))
    return {
        "cx": cx, "cy": cy,
        "rx": float(rx), "ry": float(ry),
        "color": np.random.rand(3).tolist(),
    }


def _gen_circle():
    rx, ry = random_radius()
    cx = float(np.random.uniform(-(1 + rx), 1 + rx))
    cy = float(np.random.uniform(-(1 + ry), 1 + ry))
    return {
        "cx": cx, "cy": cy,
        "rx": float(rx), "ry": float(ry),
        "color": np.random.rand(3).tolist(),
    }


# ── Unary functions (arity 1) ─────────────────────────────────────────────────


def sigmoid(x):
    return 1 / (1 + np.exp(-(x - 0.5) * 6))


def mirrored_sigmoid(x):
    return 1 / (1 + np.exp(x))


def absolute_value(x):
    return np.abs(x)


def sharpen(x):
    return separable_sharpen(x)


def blur(x):
    return separable_blur(x)


def color_rotate(x, angles=None):
    if angles is None:
        angles = (np.random.rand(3) * 2 * np.pi).tolist()
    if x.shape[-1] == 3:
        rot = R.from_euler("zyx", angles)
        return rot.apply(x.reshape(-1, 3)).reshape(x.shape)
    return np.random.rand(*x.shape)


def _gen_color_rotate():
    return {"angles": (np.random.rand(3) * 2 * np.pi).tolist()}


def kaleidoscope(x, n=None, phase=None):
    """Fold the image into ``n`` mirrored wedges around the centre.

    Expressed as a backward map (source position ``(r cos t, -r sin t)`` for
    folded angle ``t``) rather than a nearest-neighbour lookup in (angle,
    radius) space.  The old form built a ``NearestNDInterpolator`` KD-tree per
    frame per channel, which is why the README warned against sizes above
    400x400; this is three orders of magnitude cheaper and the size limit is
    gone.  The two differ only very near the centre, where nearest-in-(angle,
    radius) and nearest-in-(x, y) disagree.
    """
    if not is_valid_shape(x[0]):
        return x
    if n is None:
        p = np.array([1, 1, 0.75, 0.6, 0.45])
        p /= p.sum()
        n = int(np.random.choice([3, 5, 6, 7, 8], p=p))
    if phase is None:
        phase = float(np.random.rand() * (2 * np.pi / n))
    return warp("kaleidoscope", x, n=n, phase=phase)


def _gen_kaleidoscope():
    p = np.array([1, 1, 0.75, 0.6, 0.45])
    p /= p.sum()
    n = int(np.random.choice([3, 5, 6, 7, 8], p=p))
    phi = 2 * np.pi / n
    return {"n": n, "phase": float(np.random.rand() * phi)}


# ── Unary warp: swirl ────────────────────────────────────────────────────────


def swirl(x, cx=None, cy=None, strength=None, power=None):
    if not is_valid_shape(x[0]):
        return x
    if cx is None:
        cx = float(np.random.uniform(-0.5, 0.5))
    if cy is None:
        cy = float(np.random.uniform(-0.5, 0.5))
    if strength is None:
        strength = float(np.random.choice([-1, 1]) * np.random.uniform(0.3, np.pi))
    if power is None:
        power = -2.0
    return warp("swirl", x, cx=cx, cy=cy, strength=strength, power=power)


def _gen_swirl():
    return {
        "cx": float(np.random.uniform(-0.5, 0.5)),
        "cy": float(np.random.uniform(-0.5, 0.5)),
        "strength": float(np.random.choice([-1, 1]) * np.random.uniform(0.3, np.pi)),
        "power": float(np.random.choice([-2.0, -1.0, 1.0, 2.0])),
    }


# ── Unary warp: ripple ───────────────────────────────────────────────────────


def ripple(x, ax=None, ay=None, kx=None, ky=None, phase_x=None, phase_y=None):
    if not is_valid_shape(x[0]):
        return x
    if ax is None:
        ax = float(np.random.uniform(0.05, 0.3))
    if ay is None:
        ay = float(np.random.uniform(0.05, 0.3))
    if kx is None:
        kx = float(np.random.uniform(2.0, 8.0))
    if ky is None:
        ky = float(np.random.uniform(2.0, 8.0))
    if phase_x is None:
        phase_x = float(np.random.uniform(0.0, 2 * np.pi))
    if phase_y is None:
        phase_y = float(np.random.uniform(0.0, 2 * np.pi))
    return warp("ripple", x, ax=ax, ay=ay, kx=kx, ky=ky,
                phase_x=phase_x, phase_y=phase_y)


def _gen_ripple():
    return {
        "ax": float(np.random.uniform(0.05, 0.3)),
        "ay": float(np.random.uniform(0.05, 0.3)),
        "kx": float(np.random.uniform(2.0, 8.0)),
        "ky": float(np.random.uniform(2.0, 8.0)),
        "phase_x": float(np.random.uniform(0.0, 2 * np.pi)),
        "phase_y": float(np.random.uniform(0.0, 2 * np.pi)),
    }


# ── Unary warp: pinch/bulge ───────────────────────────────────────────────────


def pinch(x, cx=None, cy=None, strength=None):
    if not is_valid_shape(x[0]):
        return x
    if cx is None:
        cx = float(np.random.uniform(-0.3, 0.3))
    if cy is None:
        cy = float(np.random.uniform(-0.3, 0.3))
    if strength is None:
        strength = float(np.random.choice([-1, 1]) * np.random.uniform(0.2, 0.8))
    return warp("pinch", x, cx=cx, cy=cy, strength=strength)


def _gen_pinch():
    return {
        "cx": float(np.random.uniform(-0.3, 0.3)),
        "cy": float(np.random.uniform(-0.3, 0.3)),
        "strength": float(np.random.choice([-1, 1]) * np.random.uniform(0.2, 0.8)),
    }


# ── Unary warp: polar remap ───────────────────────────────────────────────────


def polar_warp(x, cx=None, cy=None):
    if not is_valid_shape(x[0]):
        return x
    if cx is None:
        cx = float(np.random.uniform(-0.3, 0.3))
    if cy is None:
        cy = float(np.random.uniform(-0.3, 0.3))
    return warp("polar_warp", x, cx=cx, cy=cy)


def _gen_polar_warp():
    return {
        "cx": float(np.random.uniform(-0.3, 0.3)),
        "cy": float(np.random.uniform(-0.3, 0.3)),
    }


# ── Binary functions (arity 2) ────────────────────────────────────────────────


def saddle(a, b):
    return a**2 - b**2


def safe_divide(a, b, eps=1e-3):
    xp = cp.get_array_module(a) if cp is not None else np
    with np.errstate(divide='ignore', invalid='ignore'):
        out = a / b
    out = xp.where(xp.isinf(out), xp.sign(b) * np.float32(1 / eps), out)
    out = xp.where(xp.isnan(out), np.float32(0.0), out)
    return out


def safe_modulus(a, b, eps=1e-10):
    return np.mod(a, np.where(b == 0, np.float32(eps), b))


def swap_phase_amplitude(a, b, axes=(1, 2)):
    fft_a = np.fft.fft2(a, axes=axes)
    fft_b = np.fft.fft2(b, axes=axes)
    phi_a = np.arctan2(np.imag(fft_a), np.real(fft_a))
    swapped_a = np.abs(fft_b) * (np.cos(phi_a) + np.sin(phi_a) * 1j)
    return np.abs(np.fft.ifft2(swapped_a, axes=axes)).astype(np.float32)


# ── Ternary functions (arity 3) ──────────────────────────────────────────────

_LUMA_WEIGHTS = np.array([0.2126, 0.7152, 0.0722], dtype=np.float32)


def _luminance(img):
    """Collapse (..., 3) → (...) using BT.709 luminance weights."""
    return (img * _LUMA_WEIGHTS).sum(axis=-1)


def blend(a, b, mask):
    m = np.clip(mask, 0.0, 1.0)
    return a * m + b * (1.0 - m)


def rgb_compose(a, b, c):
    out = np.empty_like(a)
    out[..., 0] = _luminance(a)
    out[..., 1] = _luminance(b)
    out[..., 2] = _luminance(c)
    return out


def warp_by(img, dx_field, dy_field, amplitude=0.3):
    if cp is not None:
        if isinstance(img, cp.ndarray):
            img = cp.asnumpy(img)
        if isinstance(dx_field, cp.ndarray):
            dx_field = cp.asnumpy(dx_field)
        if isinstance(dy_field, cp.ndarray):
            dy_field = cp.asnumpy(dy_field)
    chunk_size, dy_size, dx_size, _ = img.shape
    xs, ys = linear_mesh(dx=dx_size, dy=dy_size)
    frames = []
    for k in range(chunk_size):
        frame = img[k]
        x_src = xs + amplitude * _luminance(dx_field[k])
        y_src = ys + amplitude * _luminance(dy_field[k])
        col_src = (x_src + 1) / 2 * (dx_size - 1)
        row_src = (y_src + 1) / 2 * (dy_size - 1)
        frames.append(np.stack(
            [map_coordinates(frame[:, :, c], [row_src, col_src], order=1, mode="nearest")
             for c in range(3)],
            axis=-1,
        ).astype(np.float32))
    return np.stack(frames, axis=0)


def _gen_warp_by():
    return {"amplitude": float(np.random.uniform(0.1, 0.5))}


# ── Binary warp: hsv_warp ─────────────────────────────────────────────────────


def _hsv_warp_frame(image, field_frame, amplitude):
    dy, dx, _ = image.shape
    xs, ys = linear_mesh(dx=dx, dy=dy)
    hsv = rgb_to_hsv(np.clip(field_frame, 0.0, 1.0))
    h, s, v = hsv[..., 0], hsv[..., 1], hsv[..., 2]
    angle = h * (2.0 * np.pi)
    dist = s * v * amplitude
    x_src = xs + dist * np.cos(angle)
    y_src = ys + dist * np.sin(angle)
    col_src = (x_src + 1) / 2 * (dx - 1)
    row_src = (y_src + 1) / 2 * (dy - 1)
    return np.stack(
        [map_coordinates(image[:, :, c], [row_src, col_src], order=1, mode="nearest")
         for c in range(3)],
        axis=-1,
    ).astype(np.float32)


def hsv_warp(img, field, amplitude=0.3):
    if cp is not None:
        if isinstance(img, cp.ndarray):
            img = cp.asnumpy(img)
        if isinstance(field, cp.ndarray):
            field = cp.asnumpy(field)
    return np.stack(
        [_hsv_warp_frame(img[k], field[k], amplitude) for k in range(img.shape[0])],
        axis=0,
    )


def _gen_hsv_warp():
    return {"amplitude": float(np.random.uniform(0.1, 0.5))}


# ── Circular hue functions (arity 2, HSV H-channel) ──────────────────────────


def circular_mean(h1, h2):
    """Mean of two hue values along the shorter arc."""
    a1, a2 = h1 * (2 * np.pi), h2 * (2 * np.pi)
    return (np.arctan2(np.sin(a1) + np.sin(a2), np.cos(a1) + np.cos(a2)) / (2 * np.pi)) % 1.0


def circular_mean_far(h1, h2):
    """Mean of two hue values along the longer arc."""
    return (circular_mean(h1, h2) + 0.5) % 1.0


def hue_diff(h1, h2):
    """Angular distance between two hue values, in [0, 0.5]."""
    diff = np.abs(h1 - h2) % 1.0
    return np.minimum(diff, 1.0 - diff)


def hue_rotate(h1, h2):
    """Rotate h1 by h2 (circular addition)."""
    return (h1 + h2) % 1.0
