import numpy as np
from scipy.ndimage import convolve
from scipy.interpolate import NearestNDInterpolator
from scipy.spatial.transform import Rotation as R

# Adaptor functions for the recursive generator
# Note: using python's random module because numpy's doesn't handle seeds longer than 32 bits.
def rand_color(dx=None, dy=None):
    return np.random.rand(1, 1, 3)

def x_var(dx=None, dy=None):
    x_array = np.linspace(0., 1., dx).reshape(1, -1, 1)
    return x_array

def y_var(dx=None, dy=None):
    y_array = np.linspace(0., 1., dy).reshape(-1, 1, 1)
    return y_array

def linear_mesh(dx=None, dy=None):
    y = np.repeat(np.linspace(-1, 1, dy).reshape(-1, 1), dx, axis=1)
    x = np.repeat(np.linspace(-1, 1, dx).reshape(1, -1), dy, axis=0)
    return x, y

def random_point():
    return (1 - np.random.rand(2)**2)*4 - 2

def random_radius(ellipsoid=True):
    size = 2 if ellipsoid else 1
    radius = 1 - np.random.rand(size)**2
    return radius

def cone(dx=None, dy=None, ellipsoid=True):
    cx, cy = random_point()
    x, y = linear_mesh(dx=dx, dy=dy)
    rx, ry = random_radius()
    ellipsoid = np.sqrt(((x - cx)/ rx) ** 2 + ((y - cy)/ ry)**2).reshape(dy, dx, 1)
    return ellipsoid

def circle(ellipsoid=True, dx=None, dy=None):
    base = cone(dx, dy, ellipsoid).squeeze()
    circ = np.ones((dy, dx, 3)) * rand_color()
    circ[base > 1] = 0
    return circ

def saddle(a, b):
    return a**2 - b**2

def circular_mean(h1, h2):
    """Mean of two hue values along the shorter arc."""
    a1, a2 = h1 * (2 * np.pi), h2 * (2 * np.pi)
    return np.arctan2(np.sin(a1) + np.sin(a2), np.cos(a1) + np.cos(a2)) / (2 * np.pi) % 1.0

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

def hsv_to_rgb(hsv):
    """Convert (..., 3) HSV array (H in [0,1]) to RGB."""
    h, s, v = hsv[..., 0], hsv[..., 1], hsv[..., 2]
    i = (h * 6).astype(np.int32) % 6
    f = h * 6 - np.floor(h * 6)
    p, q, t = v * (1 - s), v * (1 - f * s), v * (1 - (1 - f) * s)
    rgb = np.zeros_like(hsv)
    for sector, (r, g, b) in enumerate([(v,t,p),(q,v,p),(p,v,t),(p,q,v),(t,p,v),(v,p,q)]):
        mask = i == sector
        rgb[..., 0] = np.where(mask, r, rgb[..., 0])
        rgb[..., 1] = np.where(mask, g, rgb[..., 1])
        rgb[..., 2] = np.where(mask, b, rgb[..., 2])
    return rgb

def safe_divide(a, b, eps=1e-3):
    return a / np.where(np.abs(b) < eps, np.float32(eps), b)

def safe_modulus(a, b, eps=1e-10):
    return np.mod(a, np.where(b == 0, np.float32(eps), b))

def sigmoid(x):
    x = (x - .5)*6
    return 1 / (1 + np.exp(-x))

def mirrored_sigmoid(x):
    return 1 / (1 + np.exp(x))

def absolute_value(x):
    return np.abs(x)

def color_rotate(x):
    if x.shape[-1] == 3:
        rot = R.from_euler("zyx", np.random.rand(3)*2*np.pi)
        return rot.apply(x.reshape(-1, 3)).reshape(x.shape)
    else:
        return np.random.rand(*x.shape)

gaussian_kernel_5 = np.array([[1, 4, 6, 4, 1],
                              [4, 16, 24, 16, 4],
                              [6, 24, 36, 24, 6],
                              [4, 16, 24, 16, 4],
                              [1, 4, 6, 4, 1]])/256
gaussian_kernel_3 = np.array([[1, 2, 1],
                              [2, 4, 2],
                              [1, 2, 1]])/16
sharpen_kernel_5 = np.array([[1, 4, 6, 4, 1],
                             [4, 16, 24, 16, 4],
                             [6, 24, -476, 24, 6],
                             [4, 16, 24, 16, 4],
                             [1, 4, 6, 4, 1]])*-1/256
sharpen_kernel_3 = np.array([[0, -1, 0],
                             [-1, 5, -1],
                             [0, -1, 0]])

def _apply_kernel(frame, kernel):
    channels = [np.expand_dims(convolve(c, kernel), 2) for c in frame.transpose(2, 0, 1)]
    return np.concatenate(channels, axis=2)

def sharpen(x):
    return np.stack([_apply_kernel(x[i], sharpen_kernel_5) for i in range(x.shape[0])])

def blur(x):
    return np.stack([_apply_kernel(x[i], gaussian_kernel_5) for i in range(x.shape[0])])

def swap_phase_amplitude(a, b, axes=[1, 2]):
    fft_a = np.fft.fft2(a, axes=axes)
    fft_b = np.fft.fft2(b, axes=axes)
    abs_b = np.abs(fft_b)
    phi_a = np.arctan2(np.imag(fft_a), np.real(fft_a))
    swapped_a = abs_b*(np.cos(phi_a) + np.sin(phi_a)*1j)
    return np.abs(np.fft.ifft2(swapped_a, axes=axes)).astype(np.float32)

def get_radius(x, y):
    return np.sqrt(x**2 + y**2)

def is_valid_shape(image):
    if image.ndim != 3:
        return False
    dy, dx, channels = image.shape
    if dy < 3 or dx < 3 or channels != 3:
        return False
    return True

def _kaleidoscope_frame(image, points, new_points):
    dx, dy, _ = image.shape
    interp_funcs = [NearestNDInterpolator(points, image[:, :, c].flatten()) for c in range(3)]
    new_channels = np.concatenate([f(new_points).reshape(-1, 1) for f in interp_funcs], axis=1)
    return new_channels.reshape(dx, dy, 3).astype(np.float32)

def kaleidoscope(x):
    image = x[0]
    if not is_valid_shape(image):
        return x
    p = np.array([1, 1, .75, .6, .45])
    p /= p.sum()
    n = np.random.choice([3, 5, 6, 7, 8], p=p)
    phi = 2 * np.pi / n
    phase = np.random.rand() * phi

    dx, dy, _ = image.shape
    angles = np.arctan2(-linear_mesh(dy, dx)[1], linear_mesh(dy, dx)[0])
    angles[angles < 0] += 2 * np.pi
    radiuses = get_radius(*linear_mesh(dy, dx))
    points = np.concatenate([angles.reshape(-1, 1), radiuses.reshape(-1, 1)], axis=1)
    new_points = np.concatenate([(angles % phi + phase).reshape(-1, 1), radiuses.reshape(-1, 1)], axis=1)

    return np.stack([_kaleidoscope_frame(x[i], points, new_points) for i in range(x.shape[0])])

BUILD_FUNCTIONS = ((0, rand_color),
                   (0, x_var),
                   (0, y_var),
                   (0, circle),
                   (0, cone),

                   (1, np.sin),
                   (1, np.cos),
                   (1, sigmoid),
                   (1, mirrored_sigmoid),
                   (1, sharpen),
                   (1, blur),
                   (1, absolute_value),
                   (1, color_rotate),
                   (1, kaleidoscope),

                   (2, np.add),
                   (2, np.subtract),
                   (2, np.multiply),
                   (2, safe_divide),
                   (2, safe_modulus),
                   (2, saddle),
                   (2, swap_phase_amplitude),
                   (2, circular_mean),
                   (2, circular_mean_far),
                   (2, hue_diff),
                   (2, hue_rotate))

BUILD_FUNCTIONS = sorted(BUILD_FUNCTIONS, key=lambda x: x[1].__name__)
                   # 12 functions
