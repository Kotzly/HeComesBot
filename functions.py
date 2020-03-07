import numpy as np
from config import get_config

dx, dy = get_config("dims")

_x_array = np.linspace(0., 1., dx).reshape(1, -1, 1)
_y_array = np.linspace(0., 1., dy).reshape(-1, 1, 1)

# Adaptor functions for the recursive generator
# Note: using python's random module because numpy's doesn't handle seeds longer than 32 bits.
def randColor():
    return np.random.rand(1, 1, 3)

def x_var():
    return _x_array

def y_var():
    return _y_array

def circle():
    y = np.repeat(np.linspace(-1, 1, dy).reshape(-1, 1), dx, axis=1)/2
    x = np.repeat(np.linspace(-1, 1, dx).reshape(1, -1), dy, axis=0)/2
    r = 1 - np.random.rand()**2
    cx, cy = np.random.rand(2)
    mask = np.sqrt((x - cx)**2 + (y - cy)**2) > r
    circ = np.ones((dy, dx, 3))*randColor()
    circ[mask] = 0
    return circ

def saddle(a, b):
    return a**2 - b**2

def safe_divide(a, b, eps=1e-3):
    small_values = (np.abs(b) < eps)
    b[small_values] = np.sign(b[small_values])*eps
    b[b==0] = eps
    return np.divide(a, b)

def sigmoid(x):
    return 1 / (1 + np.exp(-x))

def mirrored_sigmoid(x):
    return 1 / (1 + np.exp(x))

BUILD_FUNCTIONS = ((0, randColor),
                   (0, x_var),
                   (0, y_var),
                   (0, circle),

                   (1, np.sin),
                   (1, np.cos),
                   (1, sigmoid),
                   (1, mirrored_sigmoid),
                   
                   (2, np.add),
                   (2, np.subtract),
                   (2, np.multiply),
                   (2, safe_divide))
                   # 12 functions
