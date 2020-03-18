import numpy as np
from config import get_config
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

def safe_divide(a, b, eps=1e-3):
    small_values = (np.abs(b) < eps)
    b[small_values] = np.sign(b[small_values])*eps
    b[b==0] = eps
    return np.divide(a, b)

def safe_modulus(a, b, eps=1e-10):
    b[b==0] = eps
    modulus =  np.mod(a, b)
    return modulus

def sigmoid(x):
    x = (x - .5)*6
    return 1 / (1 + np.exp(-x))

def mirrored_sigmoid(x):
    return 1 / (1 + np.exp(x))    

def absolute_value(x):
    return np.abs(x)

def color_rotate(x):
    if x.shape[-1] == 3:
        flattened = x.reshape(-1, 3)
        rotated = R.from_euler("zyx", np.random.rand(3)*2*np.pi).apply(flattened)
        reshaped = rotated.reshape(*x.shape)
    else:
        x = np.random.rand(*x.shape)
        reshaped = x
    return reshaped

gaussian_kernel_5 = np.array([[1, 4, 6, 4, 6],
                              [4, 16, 24, 16, 4],
                              [6, 24, 36, 24, 6],
                              [4, 16, 24, 16, 4],
                              [1, 4, 6, 4, 1]])/256
gaussian_kernel_3 = np.array([[1, 2, 1],
                              [2, 4, 2],
                              [1, 2, 1]])/16
sharpen_kernel_5 = np.array([[1, 4, 6, 4, 6],
                             [4, 16, 24, 16, 4],
                             [6, 24, -476, 24, 6],
                             [4, 16, 24, 16, 4],
                             [1, 4, 6, 4, 1]])*-1/256
sharpen_kernel_3 = np.array([[0, -1, 0],
                             [-1, 5, -1],
                             [0, -1, 0]])

def sharpen(x):
    channels = [np.expand_dims(convolve(c, sharpen_kernel_5), 2) for c in x.transpose(2, 0, 1)]
    sharpened = np.concatenate(channels, axis=2)
    return sharpened

def blur(x):
    channels = [np.expand_dims(convolve(c, gaussian_kernel_5), 2) for c in x.transpose(2, 0, 1)]
    blurred = np.concatenate(channels, axis=2)
    return blurred

def swap_phase_amplitude(a, b, axes=[0, 1]):
    fft_a = np.fft.fft2(a, axes=axes)
    fft_b = np.fft.fft2(b, axes=axes)
    abs_a = np.abs(fft_a)
    abs_b = np.abs(fft_b)
    phi_a = np.arctan2(np.imag(fft_a), np.real(fft_a))
    phi_b = np.arctan2(np.imag(fft_b), np.real(fft_b))
    swapped_a = abs_b*(np.cos(phi_a) + np.sin(phi_a)*1j)
    swapped_b = abs_a*(np.cos(phi_b) + np.sin(phi_b)*1j)
    output_a = np.abs(np.fft.ifft2(swapped_a, axes=axes)).astype(np.uint8)
    output_b = np.abs(np.fft.ifft2(swapped_b, axes=axes)).astype(np.uint8)
    return output_a#, output_bnterp

def get_angle(x, y, dx=None, dy=None):
    return np.arctan2(dx/2 - x, dy/2 - y)

def get_radius(x, y):
    return np.sqrt(x**2 + y**2)

def is_valid_shape(image):
    if image.ndim != 3:
        return False
    dy, dx, channels = image.shape
    if dy < 3 or dx < 3 or channels != 3:
        return False
    return True

def kaleidoscope(image, n=3, phase=0):
    if not is_valid_shape(image):
        return image
    start = time.time()
    p = np.array([1, 1, .75, .6, .45])
    p /= p.sum()
    n = np.random.choice([3, 5, 6, 7, 8], p=p)
    phi = 2 * np.pi / n
    phase= np.random.rand()*phi

    dx, dy, _ = image.shape
    mesh_x, mesh_y = linear_mesh(dy, dx)
    angles = np.arctan2(- mesh_y, mesh_x)
    angles[angles < 0] = angles[angles < 0] + 2*np.pi
    radiuses = get_radius(*linear_mesh(dy, dx))
    points = np.concatenate([angles.reshape(-1, 1), radiuses.reshape(-1, 1)], axis=1)
    def interp(x):
        interp_funcs = [NearestNDInterpolator(points, image[:, :, c].flatten()) for c in range(3)]
        new_channels = np.concatenate([f(x).reshape(-1, 1) for f in interp_funcs], axis=1)
        return new_channels
    new_angles = (angles % phi) + phase
    new_points = np.concatenate([new_angles.reshape(-1, 1), radiuses.reshape(-1, 1)], axis=1)
    new_image = interp(new_points).reshape(dx, dy, 3).astype(np.uint8)
    return new_image

BUILD_FUNCTIONS = ((0, rand_color),
                   (0, x_var),
                   (0, y_var),
                   (0, circle),
                   (0, cone),

                   (1, np.sin),
                   (1, np.cos),
                   (1, sigmoid),
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
