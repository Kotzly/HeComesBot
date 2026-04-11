import numpy as np
from scipy.ndimage import convolve

# ── Utilities ─────────────────────────────────────────────────────────────────


def linear_mesh(dx=None, dy=None):
    y = np.repeat(np.linspace(-1, 1, dy).reshape(-1, 1), dx, axis=1)
    x = np.repeat(np.linspace(-1, 1, dx).reshape(1, -1), dy, axis=0)
    return x, y


def random_point():
    return (1 - np.random.rand(2) ** 2) * 4 - 2


def random_radius():
    return 1 - np.random.rand(2) ** 2


def get_radius(x, y):
    return np.sqrt(x**2 + y**2)


def is_valid_shape(image):
    if image.ndim != 3:
        return False
    dy, dx, channels = image.shape
    return dy >= 3 and dx >= 3 and channels == 3


def rgb_to_hsv(rgb):
    """Convert (..., 3) RGB array in [0, 1] to HSV."""
    r, g, b = rgb[..., 0], rgb[..., 1], rgb[..., 2]
    cmax = np.maximum(np.maximum(r, g), b)
    cmin = np.minimum(np.minimum(r, g), b)
    delta = cmax - cmin
    # Hue
    h = np.zeros_like(cmax)
    mask_r = (cmax == r) & (delta > 0)
    mask_g = (cmax == g) & (delta > 0)
    mask_b = (cmax == b) & (delta > 0)
    h[mask_r] = ((g[mask_r] - b[mask_r]) / delta[mask_r]) % 6.0
    h[mask_g] = (b[mask_g] - r[mask_g]) / delta[mask_g] + 2.0
    h[mask_b] = (r[mask_b] - g[mask_b]) / delta[mask_b] + 4.0
    h = h / 6.0
    # Saturation
    s = np.where(cmax > 0, delta / cmax, 0.0)
    return np.stack([h, s, cmax], axis=-1).astype(np.float32)


def hsv_to_rgb(hsv):
    """Convert (..., 3) HSV array in [0, 1] to RGB."""
    h, s, v = hsv[..., 0], hsv[..., 1], hsv[..., 2]
    i = (h * 6).astype(np.int32)
    f = h * 6 - i
    p = v * (1 - s)
    q = v * (1 - f * s)
    t = v * (1 - (1 - f) * s)
    i6 = i % 6
    r = np.select(
        [i6 == 0, i6 == 1, i6 == 2, i6 == 3, i6 == 4, i6 == 5], [v, q, p, p, t, v]
    )
    g = np.select(
        [i6 == 0, i6 == 1, i6 == 2, i6 == 3, i6 == 4, i6 == 5], [t, v, v, q, p, p]
    )
    b = np.select(
        [i6 == 0, i6 == 1, i6 == 2, i6 == 3, i6 == 4, i6 == 5], [p, p, t, v, v, q]
    )
    return np.stack([r, g, b], axis=-1).astype(np.float32)


_gaussian_kernel_5 = (
    np.array(
        [
            [1, 4, 6, 4, 1],
            [4, 16, 24, 16, 4],
            [6, 24, 36, 24, 6],
            [4, 16, 24, 16, 4],
            [1, 4, 6, 4, 1],
        ]
    )
    / 256
)

_sharpen_kernel_5 = (
    np.array(
        [
            [1, 4, 6, 4, 1],
            [4, 16, 24, 16, 4],
            [6, 24, -476, 24, 6],
            [4, 16, 24, 16, 4],
            [1, 4, 6, 4, 1],
        ]
    )
    * -1
    / 256
)


def _apply_kernel(frame, kernel):
    channels = [
        np.expand_dims(convolve(c, kernel), 2) for c in frame.transpose(2, 0, 1)
    ]
    return np.concatenate(channels, axis=2)
