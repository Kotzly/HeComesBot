import numpy as np
from dataclasses import dataclass, field
from typing import Callable, Optional
from scipy.ndimage import convolve
from scipy.interpolate import NearestNDInterpolator
from scipy.spatial.transform import Rotation as R


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
    return np.sqrt(x ** 2 + y ** 2)

def is_valid_shape(image):
    if image.ndim != 3:
        return False
    dy, dx, channels = image.shape
    return dy >= 3 and dx >= 3 and channels == 3

def hsv_to_rgb(hsv):
    """Convert (..., 3) HSV array in [0, 1] to RGB."""
    h, s, v = hsv[..., 0], hsv[..., 1], hsv[..., 2]
    i  = (h * 6).astype(np.int32)
    f  = h * 6 - i
    p  = v * (1 - s)
    q  = v * (1 - f * s)
    t  = v * (1 - (1 - f) * s)
    i6 = i % 6
    r = np.select([i6==0, i6==1, i6==2, i6==3, i6==4, i6==5], [v, q, p, p, t, v])
    g = np.select([i6==0, i6==1, i6==2, i6==3, i6==4, i6==5], [t, v, v, q, p, p])
    b = np.select([i6==0, i6==1, i6==2, i6==3, i6==4, i6==5], [p, p, t, v, v, q])
    return np.stack([r, g, b], axis=-1).astype(np.float32)

_gaussian_kernel_5 = np.array([[1, 4, 6, 4, 1],
                                [4, 16, 24, 16, 4],
                                [6, 24, 36, 24, 6],
                                [4, 16, 24, 16, 4],
                                [1, 4, 6, 4, 1]]) / 256

_sharpen_kernel_5  = np.array([[1, 4, 6, 4, 1],
                                [4, 16, 24, 16, 4],
                                [6, 24, -476, 24, 6],
                                [4, 16, 24, 16, 4],
                                [1, 4, 6, 4, 1]]) * -1 / 256

def _apply_kernel(frame, kernel):
    channels = [np.expand_dims(convolve(c, kernel), 2) for c in frame.transpose(2, 0, 1)]
    return np.concatenate(channels, axis=2)


# ── Leaf functions (arity 0) ──────────────────────────────────────────────────

def rand_color(dx=None, dy=None, color=None):
    if color is None:
        color = np.random.rand(3).astype(np.float32)
    else:
        color = np.asarray(color, dtype=np.float32)
    return np.broadcast_to(color.reshape(1, 1, 3), (dy, dx, 3)).copy()

def _gen_rand_color():
    return {'color': np.random.rand(3).tolist()}


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
    return {'angle': float(np.random.rand() * 2 * np.pi),
            'color': np.random.rand(3).tolist()}


def cone(dx=None, dy=None, cx=None, cy=None, rx=None, ry=None):
    if cx is None or cy is None:
        cx, cy = random_point()
    if rx is None or ry is None:
        rx, ry = random_radius()
    x, y = linear_mesh(dx=dx, dy=dy)
    gradient = np.sqrt(((x - cx) / rx) ** 2 + ((y - cy) / ry) ** 2).reshape(dy, dx, 1).astype(np.float32)
    return np.broadcast_to(gradient, (dy, dx, 3)).copy()

def _gen_cone():
    cx, cy = random_point()
    rx, ry = random_radius()
    return {'cx': float(cx), 'cy': float(cy), 'rx': float(rx), 'ry': float(ry)}


def circle(dx=None, dy=None, cx=None, cy=None, rx=None, ry=None, color=None):
    base = cone(dx=dx, dy=dy, cx=cx, cy=cy, rx=rx, ry=ry)
    if color is None:
        color = np.random.rand(3).astype(np.float32)
    else:
        color = np.asarray(color, dtype=np.float32)
    circ = np.ones((dy, dx, 3), dtype=np.float32) * color.reshape(1, 1, 3)
    circ[base > 1] = 0
    return circ

def _gen_circle():
    cx, cy = random_point()
    rx, ry = random_radius()
    return {'cx': float(cx), 'cy': float(cy), 'rx': float(rx), 'ry': float(ry),
            'color': np.random.rand(3).tolist()}


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
    return np.stack([_apply_kernel(x[i], _gaussian_kernel_5) for i in range(x.shape[0])])


def color_rotate(x, angles=None):
    if angles is None:
        angles = (np.random.rand(3) * 2 * np.pi).tolist()
    if x.shape[-1] == 3:
        rot = R.from_euler("zyx", angles)
        return rot.apply(x.reshape(-1, 3)).reshape(x.shape)
    return np.random.rand(*x.shape)

def _gen_color_rotate():
    return {'angles': (np.random.rand(3) * 2 * np.pi).tolist()}


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
    angles   = np.arctan2(-linear_mesh(dy, dx)[1], linear_mesh(dy, dx)[0])
    angles[angles < 0] += 2 * np.pi
    radiuses = get_radius(*linear_mesh(dy, dx))
    points     = np.concatenate([angles.reshape(-1, 1),              radiuses.reshape(-1, 1)], axis=1)
    new_points = np.concatenate([(angles % phi + phase).reshape(-1, 1), radiuses.reshape(-1, 1)], axis=1)
    return np.stack([_kaleidoscope_frame(x[i], points, new_points) for i in range(x.shape[0])])

def _gen_kaleidoscope():
    p = np.array([1, 1, .75, .6, .45])
    p /= p.sum()
    n   = int(np.random.choice([3, 5, 6, 7, 8], p=p))
    phi = 2 * np.pi / n
    return {'n': n, 'phase': float(np.random.rand() * phi)}


# ── Binary functions (arity 2) ────────────────────────────────────────────────

def saddle(a, b):
    return a ** 2 - b ** 2

def safe_divide(a, b, eps=1e-3):
    return a / np.where(np.abs(b) < eps, np.float32(eps), b)

def safe_modulus(a, b, eps=1e-10):
    return np.mod(a, np.where(b == 0, np.float32(eps), b))

def swap_phase_amplitude(a, b, axes=(1, 2)):
    fft_a     = np.fft.fft2(a, axes=axes)
    fft_b     = np.fft.fft2(b, axes=axes)
    phi_a     = np.arctan2(np.imag(fft_a), np.real(fft_a))
    swapped_a = np.abs(fft_b) * (np.cos(phi_a) + np.sin(phi_a) * 1j)
    return np.abs(np.fft.ifft2(swapped_a, axes=axes)).astype(np.float32)


# ── Registry ──────────────────────────────────────────────────────────────────

@dataclass
class FunctionDef:
    func:     Callable
    arity:    int
    params:   list             = field(default_factory=list)
    generate: Optional[Callable] = None


FUNCTION_REGISTRY = [
    # ── Leaves (arity 0) ──────────────────────────────────────────────────────
    FunctionDef(rand_color, 0,
        params=[{'name': 'color', 'type': 'color', 'label': 'Color'}],
        generate=_gen_rand_color),

    FunctionDef(x_var, 0,
        params=[{'name': 'angle', 'type': 'float', 'min': 0.0, 'max': 6.2832, 'label': 'Angle'},
                {'name': 'color', 'type': 'color',                             'label': 'Color'}],
        generate=_gen_gradient),

    FunctionDef(y_var, 0,
        params=[{'name': 'angle', 'type': 'float', 'min': 0.0, 'max': 6.2832, 'label': 'Angle'},
                {'name': 'color', 'type': 'color',                             'label': 'Color'}],
        generate=_gen_gradient),

    FunctionDef(cone, 0,
        params=[{'name': 'cx', 'type': 'float', 'min': -2.0, 'max': 2.0, 'label': 'Center X'},
                {'name': 'cy', 'type': 'float', 'min': -2.0, 'max': 2.0, 'label': 'Center Y'},
                {'name': 'rx', 'type': 'float', 'min': 0.01, 'max': 1.0, 'label': 'Radius X'},
                {'name': 'ry', 'type': 'float', 'min': 0.01, 'max': 1.0, 'label': 'Radius Y'}],
        generate=_gen_cone),

    FunctionDef(circle, 0,
        params=[{'name': 'cx',    'type': 'float', 'min': -2.0, 'max': 2.0, 'label': 'Center X'},
                {'name': 'cy',    'type': 'float', 'min': -2.0, 'max': 2.0, 'label': 'Center Y'},
                {'name': 'rx',    'type': 'float', 'min': 0.01, 'max': 1.0, 'label': 'Radius X'},
                {'name': 'ry',    'type': 'float', 'min': 0.01, 'max': 1.0, 'label': 'Radius Y'},
                {'name': 'color', 'type': 'color',                           'label': 'Color'}],
        generate=_gen_circle),

    # ── Unary (arity 1) ───────────────────────────────────────────────────────
    FunctionDef(np.sin,          1),
    FunctionDef(np.cos,          1),
    FunctionDef(sigmoid,         1),
    FunctionDef(mirrored_sigmoid,1),
    FunctionDef(absolute_value,  1),
    FunctionDef(sharpen,         1),
    FunctionDef(blur,            1),

    FunctionDef(color_rotate, 1,
        params=[{'name': 'angles', 'type': 'angles', 'label': 'Euler angles ZYX'}],
        generate=_gen_color_rotate),

    FunctionDef(kaleidoscope, 1,
        params=[{'name': 'n',     'type': 'int',   'choices': [3, 5, 6, 7, 8], 'label': 'Segments'},
                {'name': 'phase', 'type': 'float', 'min': 0.0, 'max': 6.2832,  'label': 'Phase'}],
        generate=_gen_kaleidoscope),

    # ── Binary (arity 2) ──────────────────────────────────────────────────────
    FunctionDef(np.add,             2),
    FunctionDef(np.subtract,        2),
    FunctionDef(np.multiply,        2),
    FunctionDef(safe_divide,        2),
    FunctionDef(safe_modulus,       2),
    FunctionDef(saddle,             2),
    FunctionDef(swap_phase_amplitude, 2),
]


# ── Derived exports ───────────────────────────────────────────────────────────

BUILD_FUNCTIONS = sorted(
    [(fd.arity, fd.func) for fd in FUNCTION_REGISTRY],
    key=lambda x: x[1].__name__,
)

_REGISTRY_BY_NAME = {fd.func.__name__: fd for fd in FUNCTION_REGISTRY}

def generate_params(func_name):
    fd = _REGISTRY_BY_NAME.get(func_name)
    if fd and fd.generate:
        return fd.generate()
    return {}

FUNC_PARAMS = {
    fd.func.__name__: fd.params
    for fd in FUNCTION_REGISTRY
    if fd.params
}
