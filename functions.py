import numpy as np
from config import get_config

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
    # cx, cy = random_point()
    # x, y = linear_mesh(dx, dy)
    # rx, ry = random_radius()
    # mask = np.sqrt(((x - cx) / rx)**2 + ((y - cy) / ry)**2) > r
    # circ = np.ones((dy, dx, 3))*rand_color()
    # circ[mask] *= 0
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

def sigmoid(x):
    x = (x - .5)*6
    return 1 / (1 + np.exp(-x))

def mirrored_sigmoid(x):
    return 1 / (1 + np.exp(x))    

BUILD_FUNCTIONS = ((0, rand_color),
                   (0, x_var),
                   (0, y_var),
                   (0, circle),
                   (0, cone),

                   (1, np.sin),
                   (1, np.cos),
                   (1, sigmoid),

                   (2, np.add),
                   (2, np.subtract),
                   (2, np.multiply),
                   (2, saddle),
                   (2, safe_divide))
                   # 12 functions
