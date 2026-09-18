import numpy as np
from scipy.ndimage import convolve, convolve1d

F32 = np.float32

# ── Utilities ─────────────────────────────────────────────────────────────────


def linear_mesh(dx=None, dy=None):
    """float32 ``[-1, 1]`` coordinate grids of shape ``(dy, dx)``.

    Returned as broadcast views, so the pair costs a few kilobytes instead of
    two materialised arrays — 4.1 MB each at 540x960, and this is rebuilt by
    every warp and every gradient leaf.  float32 also keeps warp coordinates
    from being computed at double precision and cast back down.

    The views are read-only.  Every operator here derives new arrays from them
    (``x - cx``, ``np.sqrt(...)``); none writes into the grid itself.
    """
    xs = np.linspace(-1.0, 1.0, dx, dtype=F32)[None, :]
    ys = np.linspace(-1.0, 1.0, dy, dtype=F32)[:, None]
    return np.broadcast_to(xs, (dy, dx)), np.broadcast_to(ys, (dy, dx))


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


# ── Separable form of the 5x5 kernels above ──────────────────────────────────
#
# The 5x5 binomial kernel is the outer product of [1, 4, 6, 4, 1] / 16 with
# itself, so it factors into two 1D passes: 10 multiply-adds per pixel instead
# of 25.  Applied to a whole (n, dy, dx, c) batch at once, it also replaces the
# per-frame, per-channel Python loop in _apply_kernel.  Results match the dense
# form to float32 precision.
#
# The sharpen kernel is the same numerator with its centre tap at -476 rather
# than 36, negated: that is -(gaussian - 512 * delta) / 256, i.e. 2*a - blur(a).

_BINOMIAL_5 = np.array([1, 4, 6, 4, 1], dtype=F32) / 16.0


def separable_blur(batch, output=None):
    """5x5 binomial blur over a ``(n, dy, dx, c)`` batch.

    Pass ``output=batch`` to filter in place when the input is not needed
    afterwards; ``convolve1d`` handles aliasing internally.
    """
    first = convolve1d(batch, _BINOMIAL_5, axis=1, mode="reflect", output=output)
    return convolve1d(first, _BINOMIAL_5, axis=2, mode="reflect", output=first)


def separable_sharpen(batch):
    """Unsharp mask ``2*a - blur(a)``: the separable form of the 5x5 sharpen kernel."""
    blurred = separable_blur(batch)
    return 2.0 * batch - blurred
