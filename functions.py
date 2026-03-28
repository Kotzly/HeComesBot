import numpy as np
from scipy.ndimage import convolve
from scipy.interpolate import NearestNDInterpolator
from scipy.spatial.transform import Rotation as R

# Adaptor functions for the recursive generator
# Note: using python's random module because numpy's doesn't handle seeds longer than 32 bits.
def rand_color(dx=None, dy=None):
    color = np.random.rand(1, 1, 3).astype(np.float32)
    return np.broadcast_to(color, (dy, dx, 3)).copy()

def _rotated_gradient(dx, dy, angle=None, color=None):
    """Linear gradient at a random angle, colored by a random RGB."""
    if angle is None:
        angle = np.random.rand() * 2 * np.pi
    if color is None:
        color = np.random.rand(3).astype(np.float32)
    else:
        color = np.asarray(color, dtype=np.float32)
    x, y  = linear_mesh(dx=dx, dy=dy)
    grad  = np.cos(angle) * x + np.sin(angle) * y          # (dy, dx), range subset of [-2, 2]
    return (grad[:, :, np.newaxis] * color.reshape(1, 1, 3)).astype(np.float32)

def x_var(dx=None, dy=None, angle=None, color=None):
    return _rotated_gradient(dx, dy, angle=angle, color=color)

def y_var(dx=None, dy=None, angle=None, color=None):
    return _rotated_gradient(dx, dy, angle=angle, color=color)

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

def cone(dx=None, dy=None, cx=None, cy=None, rx=None, ry=None):
    if cx is None or cy is None:
        cx, cy = random_point()
    if rx is None or ry is None:
        rx, ry = random_radius()
    x, y = linear_mesh(dx=dx, dy=dy)
    gradient = np.sqrt(((x - cx) / rx) ** 2 + ((y - cy) / ry) ** 2).reshape(dy, dx, 1).astype(np.float32)
    return np.broadcast_to(gradient, (dy, dx, 3)).copy()

def circle(dx=None, dy=None, cx=None, cy=None, rx=None, ry=None, color=None):
    base = cone(dx=dx, dy=dy, cx=cx, cy=cy, rx=rx, ry=ry)  # (dy, dx, 3), all channels equal
    if color is None:
        color = np.random.rand(3).astype(np.float32)
    else:
        color = np.asarray(color, dtype=np.float32)
    circ = np.ones((dy, dx, 3), dtype=np.float32) * color.reshape(1, 1, 3)
    circ[base > 1] = 0
    return circ

def saddle(a, b):
    return a**2 - b**2

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

def color_rotate(x, angles=None):
    if angles is None:
        angles = (np.random.rand(3) * 2 * np.pi).tolist()
    if x.shape[-1] == 3:
        rot = R.from_euler("zyx", angles)
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

def hsv_to_rgb(hsv):
    """Convert (H, S, V) image array in [0,1] to RGB. Input shape: (..., 3)."""
    h, s, v = hsv[..., 0], hsv[..., 1], hsv[..., 2]
    i = (h * 6).astype(np.int32)
    f = h * 6 - i
    p = v * (1 - s)
    q = v * (1 - f * s)
    t = v * (1 - (1 - f) * s)
    i6 = i % 6
    r = np.select([i6==0, i6==1, i6==2, i6==3, i6==4, i6==5], [v, q, p, p, t, v])
    g = np.select([i6==0, i6==1, i6==2, i6==3, i6==4, i6==5], [t, v, v, q, p, p])
    b = np.select([i6==0, i6==1, i6==2, i6==3, i6==4, i6==5], [p, p, t, v, v, q])
    return np.stack([r, g, b], axis=-1).astype(np.float32)


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

def kaleidoscope(x, n=None, phase=None):
    image = x[0]
    if not is_valid_shape(image):
        return x
    if n is None:
        p = np.array([1, 1, .75, .6, .45])
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
    new_points = np.concatenate([(angles % phi + phase).reshape(-1, 1), radiuses.reshape(-1, 1)], axis=1)

    return np.stack([_kaleidoscope_frame(x[i], points, new_points) for i in range(x.shape[0])])

def generate_params(func_name):
    """Generate random params for functions that use internal randomness."""
    if func_name in ('x_var', 'y_var'):
        return {'angle': float(np.random.rand() * 2 * np.pi),
                'color': np.random.rand(3).tolist()}
    if func_name == 'cone':
        cx, cy = random_point()
        rx, ry = random_radius()
        return {'cx': float(cx), 'cy': float(cy), 'rx': float(rx), 'ry': float(ry)}
    if func_name == 'circle':
        cx, cy = random_point()
        rx, ry = random_radius()
        color = np.random.rand(3).tolist()
        return {'cx': float(cx), 'cy': float(cy), 'rx': float(rx), 'ry': float(ry), 'color': color}
    if func_name == 'color_rotate':
        return {'angles': (np.random.rand(3) * 2 * np.pi).tolist()}
    if func_name == 'kaleidoscope':
        p = np.array([1, 1, .75, .6, .45])
        p /= p.sum()
        n = int(np.random.choice([3, 5, 6, 7, 8], p=p))
        phi = 2 * np.pi / n
        return {'n': n, 'phase': float(np.random.rand() * phi)}
    return {}


# Param specs for the web UI: maps func_name -> list of param descriptors
FUNC_PARAMS = {
    'cone': [
        {'name': 'cx', 'type': 'float', 'min': -2.0, 'max': 2.0,  'label': 'Center X'},
        {'name': 'cy', 'type': 'float', 'min': -2.0, 'max': 2.0,  'label': 'Center Y'},
        {'name': 'rx', 'type': 'float', 'min': 0.01, 'max': 1.0,  'label': 'Radius X'},
        {'name': 'ry', 'type': 'float', 'min': 0.01, 'max': 1.0,  'label': 'Radius Y'},
    ],
    'circle': [
        {'name': 'cx',    'type': 'float', 'min': -2.0, 'max': 2.0, 'label': 'Center X'},
        {'name': 'cy',    'type': 'float', 'min': -2.0, 'max': 2.0, 'label': 'Center Y'},
        {'name': 'rx',    'type': 'float', 'min': 0.01, 'max': 1.0, 'label': 'Radius X'},
        {'name': 'ry',    'type': 'float', 'min': 0.01, 'max': 1.0, 'label': 'Radius Y'},
        {'name': 'color', 'type': 'color',                          'label': 'Color'},
    ],
    'x_var': [
        {'name': 'angle', 'type': 'float', 'min': 0.0, 'max': 6.2832, 'label': 'Angle'},
        {'name': 'color', 'type': 'color',                             'label': 'Color'},
    ],
    'y_var': [
        {'name': 'angle', 'type': 'float', 'min': 0.0, 'max': 6.2832, 'label': 'Angle'},
        {'name': 'color', 'type': 'color',                             'label': 'Color'},
    ],
    'color_rotate': [
        {'name': 'angles', 'type': 'angles', 'label': 'Euler angles ZYX'},
    ],
    'kaleidoscope': [
        {'name': 'n',     'type': 'int',   'choices': [3, 5, 6, 7, 8], 'label': 'Segments'},
        {'name': 'phase', 'type': 'float', 'min': 0.0, 'max': 6.2832,  'label': 'Phase'},
    ],
}


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
                   (2, swap_phase_amplitude))

BUILD_FUNCTIONS = sorted(BUILD_FUNCTIONS, key=lambda x: x[1].__name__)
                   # 12 functions
