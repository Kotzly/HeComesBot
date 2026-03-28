import numpy as np
from scipy.interpolate import NearestNDInterpolator
from scipy.spatial.transform import Rotation as R

from hecomes.artgen.func_utils import (
    _apply_kernel,
    _gaussian_kernel_5,
    _sharpen_kernel_5,
    get_radius,
    hsv_to_rgb,
    is_valid_shape,
    linear_mesh,
    random_point,
    random_radius,
)

# ── Leaf functions (arity 0) ──────────────────────────────────────────────────


def rand_color(dx=None, dy=None, color=None):
    if color is None:
        color = np.random.rand(3).astype(np.float32)
    else:
        color = np.asarray(color, dtype=np.float32)
    return np.broadcast_to(color.reshape(1, 1, 3), (dy, dx, 3)).copy()


def _gen_rand_color():
    return {"color": np.random.rand(3).tolist()}


def _rotated_gradient(dx, dy, angle=None, color=None):
    if angle is None:
        angle = np.random.rand() * 2 * np.pi
    if color is None:
        color = np.random.rand(3).astype(np.float32)
    else:
        color = np.asarray(color, dtype=np.float32)
    x, y = linear_mesh(dx=dx, dy=dy)
    grad = np.cos(angle) * x + np.sin(angle) * y
    return (grad[:, :, np.newaxis] * color.reshape(1, 1, 3)).astype(np.float32)


def x_var(dx=None, dy=None, angle=None, color=None):
    return _rotated_gradient(dx, dy, angle=angle, color=color)


def y_var(dx=None, dy=None, angle=None, color=None):
    return _rotated_gradient(dx, dy, angle=angle, color=color)


def _gen_gradient():
    return {
        "angle": float(np.random.rand() * 2 * np.pi),
        "color": np.random.rand(3).tolist(),
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
    cx, cy = random_point()
    rx, ry = random_radius()
    return {
        "cx": float(cx), "cy": float(cy),
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
    cx, cy = random_point()
    rx, ry = random_radius()
    return {
        "cx": float(cx), "cy": float(cy),
        "rx": float(rx), "ry": float(ry),
        "color": np.random.rand(3).tolist(),
    }


def _gen_circle():
    cx, cy = random_point()
    rx, ry = random_radius()
    return {
        "cx": float(cx),
        "cy": float(cy),
        "rx": float(rx),
        "ry": float(ry),
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
    return np.stack([_apply_kernel(x[i], _sharpen_kernel_5) for i in range(x.shape[0])])


def blur(x):
    return np.stack(
        [_apply_kernel(x[i], _gaussian_kernel_5) for i in range(x.shape[0])]
    )


def color_rotate(x, angles=None):
    if angles is None:
        angles = (np.random.rand(3) * 2 * np.pi).tolist()
    if x.shape[-1] == 3:
        rot = R.from_euler("zyx", angles)
        return rot.apply(x.reshape(-1, 3)).reshape(x.shape)
    return np.random.rand(*x.shape)


def _gen_color_rotate():
    return {"angles": (np.random.rand(3) * 2 * np.pi).tolist()}


def _kaleidoscope_frame(image, points, new_points):
    dx, dy, _ = image.shape
    interp_funcs = [
        NearestNDInterpolator(points, image[:, :, c].flatten()) for c in range(3)
    ]
    new_channels = np.concatenate(
        [f(new_points).reshape(-1, 1) for f in interp_funcs], axis=1
    )
    return new_channels.reshape(dx, dy, 3).astype(np.float32)


def kaleidoscope(x, n=None, phase=None):
    image = x[0]
    if not is_valid_shape(image):
        return x
    if n is None:
        p = np.array([1, 1, 0.75, 0.6, 0.45])
        p /= p.sum()
        n = int(np.random.choice([3, 5, 6, 7, 8], p=p))
    phi = 2 * np.pi / n
    if phase is None:
        phase = float(np.random.rand() * phi)
    dx, dy, _ = image.shape
    angles = np.arctan2(-linear_mesh(dy, dx)[1], linear_mesh(dy, dx)[0])
    angles[angles < 0] += 2 * np.pi
    radiuses = get_radius(*linear_mesh(dy, dx))
    points = np.concatenate([angles.reshape(-1, 1), radiuses.reshape(-1, 1)], axis=1)
    new_points = np.concatenate(
        [(angles % phi + phase).reshape(-1, 1), radiuses.reshape(-1, 1)], axis=1
    )
    return np.stack(
        [_kaleidoscope_frame(x[i], points, new_points) for i in range(x.shape[0])]
    )


def _gen_kaleidoscope():
    p = np.array([1, 1, 0.75, 0.6, 0.45])
    p /= p.sum()
    n = int(np.random.choice([3, 5, 6, 7, 8], p=p))
    phi = 2 * np.pi / n
    return {"n": n, "phase": float(np.random.rand() * phi)}


# ── Binary functions (arity 2) ────────────────────────────────────────────────


def saddle(a, b):
    return a**2 - b**2


def safe_divide(a, b, eps=1e-3):
    return a / np.where(np.abs(b) < eps, np.float32(eps), b)


def safe_modulus(a, b, eps=1e-10):
    return np.mod(a, np.where(b == 0, np.float32(eps), b))


def swap_phase_amplitude(a, b, axes=(1, 2)):
    fft_a = np.fft.fft2(a, axes=axes)
    fft_b = np.fft.fft2(b, axes=axes)
    phi_a = np.arctan2(np.imag(fft_a), np.real(fft_a))
    swapped_a = np.abs(fft_b) * (np.cos(phi_a) + np.sin(phi_a) * 1j)
    return np.abs(np.fft.ifft2(swapped_a, axes=axes)).astype(np.float32)


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
