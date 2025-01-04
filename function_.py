import os
import sys
os.environ["TF_ENABLE_ONEDNN_OPTS"] = "0"
import math
from abc import ABC, abstractmethod
from typing import List, Tuple, Dict, Any, Optional, Union
from numpy import ndarray
from dataclasses import dataclass
from torch.nn import Module
import torch
import numpy as np
import keras
from keras.layers import Layer, Input
from keras import Model
import tensorflow as tf
import tensorflow.keras.backend as K
from tensorflow.keras import Initializer

DEFAULT_DTYPE = tf.keras.backend.floatx()


class RadiusInitializer(Initializer):

    def __init__(self, seed=None):
        self.seed = seed
        super().__init__()

    def __call__(self, shape, dtype=None):
        seed = self.seed or np.random.randint(0, sys.maxsize)
        return 1 - tf.random.uniform(shape, 0, 1, dtype=dtype, seed=seed) ** 2

    def get_config(self):  # To support serialization
        return {}


class CentralizedPointInitializer(Initializer):
        
    def __init__(self, seed=None):
        self.seed = seed
        super().__init__()

    def __call__(self, shape, dtype=None):
        seed = self.seed or np.random.randint(0, sys.maxsize)
        return (
            1 - tf.random.uniform(shape, 0, 1, dtype=dtype, seed=seed) ** 2
        ) * 4 - 2

    def get_config(self):  # To support serialization
        return {}



def get_initializer(name, args=None, seed=42):
    return dict(
        PositiveBounded=tf.keras.initializers.RandomUniform(
            *(args or (0, 1)), seed=seed
        ),
        Bounded=tf.keras.initializers.RandomUniform(*(args or (-1, 1)), seed=seed),
        Normal=tf.keras.initializers.RandomNormal(*(args or (0, 1)), seed=seed),
        PositiveNormal=tf.keras.initializers.TruncatedNormal(
            *(args or (0, 0.5)), seed=seed
        ),
        Radius=RadiusInitializer(seed=seed),
        CentralizedPoint=CentralizedPointInitializer(seed=seed),
    )[name]


def linear_mesh(width=None, height=None, channels=None, batch_dim=True, start=-1, end=1):
    x = tf.linspace(start, end, width)
    y = tf.linspace(start, end, height)
    x, y = tf.cast(x, DEFAULT_DTYPE), tf.cast(y, DEFAULT_DTYPE)
    x, y = tf.meshgrid(y, x)
    channels = channels or 1
    x = tf.expand_dims(x, -1)
    y = tf.expand_dims(y, -1)
    x = tf.tile(x, [1, 1, channels])
    y = tf.tile(y, [1, 1, channels])
    if batch_dim:
        x = tf.expand_dims(x, 0)
        y = tf.expand_dims(y, 0)
    return x, y


class BaseLayer(Layer):
    def __init__(self, width=None, height=None):
        super().__init__()
        self.width = width
        self.height = height


class BaseLeafLayer(BaseLayer):
    n_inputs = 0
    def __init__(self, width=None, height=None):
        super().__init__(width, height)
        

class Operation(BaseLayer):
    def __init__(self, width=None, height=None):
        super().__init__(width, height)


class Color(BaseLeafLayer):
    shapes =[
        (None, 1)
    ]
    def call(self, color):
        # shape is (1, 1, 1, 1) (batch, width, height, channels=1)
        color = tf.reshape(color, (-1, 1, 1, 1))
        shape = tf.constant([1,  self.width, self.height, 3], tf.int32)
        return tf.tile(color, shape)

class RGBColor(BaseLeafLayer):
    shapes =[
        (None, 3)
    ]

    def call(self, color):
        # shape is (1, 1, 1, 3) (batch, width, height, channels=3)
        color = tf.reshape(color, (-1, 1, 1, 3))
        shape = tf.constant([1,  self.width, self.height, 1], tf.int32)
        return tf.tile(color, shape)


class Cone(BaseLeafLayer):
    shapes =[
        (None, 2), # position
        (None, 1),  # radius
    ]

    def call(self, center, radius):
        
        x, y = linear_mesh(self.width, self.height, batch_dim=True)

        radius = tf.abs(radius)

        centerx = tf.reshape(center[:, 0], (-1, 1, 1, 1))
        centery = tf.reshape(center[:, 1], (-1, 1, 1, 1))
        radius = tf.reshape(radius, (-1, 1, 1, 1))

        ellipsoid = K.sqrt(
            ((x - centerx) / radius) ** 2
            + ((y - centery) / radius) ** 2
        )
        return ellipsoid


class Ellipse(BaseLeafLayer):
    shapes =[
        (None, 2), # position
        (None, 2),  # radius
    ]
    def call(self, center, radius):
        
        x, y = linear_mesh(self.width, self.height, batch_dim=True)

        radius = tf.abs(radius)

        centerx = tf.reshape(center[:, 0], (-1, 1, 1, 1))
        centery = tf.reshape(center[:, 1], (-1, 1, 1, 1))

        radiusx = tf.reshape(radius[:, 0], (-1, 1, 1, 1))
        radiusy = tf.reshape(radius[:, 1], (-1, 1, 1, 1))

        ellipsoid = K.sqrt(
            ((x - centerx) / radiusx) ** 2
            + ((y - centery) / radiusy) ** 2
        )
        ellipsoid = K.cast(ellipsoid >= 1, x.dtype)
        return ellipsoid


class Circle(BaseLeafLayer):
    shapes =[
        (None, 2), # position
        (None, 1),  # radius
    ]


    def call(self, center, radius):
        x, y = linear_mesh(self.width, self.height, batch_dim=True)

        radius = tf.abs(radius)

        centerx = tf.reshape(center[:, 0], (-1, 1, 1, 1))
        centery = tf.reshape(center[:, 1], (-1, 1, 1, 1))
        radius = tf.reshape(radius, (-1, 1, 1, 1))
        ellipsoid = K.sqrt(
            ((x - centerx) / radius) ** 2
            + ((y - centery) / radius) ** 2
        )
        ellipsoid = K.cast(ellipsoid >= 1, x.dtype)
        return ellipsoid


class SafeDivide(Operation):
    shapes = [
        (None, None, None, 3),
        (None, None, None, 3)
    ]
    n_inputs = 2

    def __init__(self, width, height, eps=1e-3):
        super().__init__(width, height)
        self.eps = eps

    def call(self, x, y):

        return tf.clip_by_value(x / y, -1 / self.eps, 1 / self.eps)


class Multiply(Operation):
    shapes = [
        (None, None, None, 3),
        (None, None, None, 3)
    ]
    n_inputs = 2

    def call(self, x, y):
        return x * y


class Gradient(BaseLeafLayer):

    shapes =[
        (None, 1), # direction
    ]

    def call(self, direction):

        v = 1 / math.sqrt(2)
        
        direction_x = tf.reshape(tf.sin(direction), (-1, 1, 1, 1))
        direction_y = tf.reshape(tf.cos(direction), (-1, 1, 1, 1))
                
        x, y = linear_mesh(self.width, self.height, channels=1, batch_dim=True, start=-v, end=v)
        img = (x * direction_x) + (y * direction_y)
        img = (img + 1) / 2
        return img
        


class RGBGradient(BaseLeafLayer):
    shapes =[
        (None, 1), # directions
        (None, 3), # color
    ]

    def call(self, direction, color):

        v = 1 / math.sqrt(2)
        
        direction_x = tf.reshape(tf.sin(direction), (-1, 1, 1, 1))
        direction_y = tf.reshape(tf.cos(direction), (-1, 1, 1, 1))
                
        color = tf.reshape(color, (-1, 1, 1, 3))
        
        x, y = linear_mesh(self.width, self.height, channels=3, batch_dim=True, start=-v, end=v)
        img = (x * direction_x) + (y * direction_y)
        img = (img + 1) / 2
        img = img * color
        return img


class Noise(BaseLeafLayer):
    n_inputs = 1
    shapes = [
        (None, 1)
    ]

    def call(self, snr):
        snr = tf.reshape(snr, (-1, 1, 1, 1))
        snr = tf.abs(snr)
        return tf.random.uniform((1, self.width, self.height, 3), 0, 1) * snr


class Sum(BaseLayer):
    shapes = [
        (None, None, None, 3),
        (None, None, None, 3)
    ]
    n_inputs = 2

    def call(self, a, b):
        return a + b


class Sub(BaseLayer):
    shapes = [
        (None, None, None, 3),
        (None, None, None, 3)
    ]
    n_inputs = 2

    def call(self, a, b):
        return a - b


class Cosine(Operation):
    shapes = [
        None
    ]
    n_inputs = 1

    def call(self, img):
        return tf.cos(img)


class Sigmoid(Operation):
    shapes = [
        None
    ]
    n_inputs = 1

    def call(self, img):
        return tf.sigmoid(img)


class Tanh(Operation):
    shapes = [
        None
    ]
    n_inputs = 1

    def call(self, img):
        return tf.tanh(img)

class Saddle(Operation):
    shapes = [
        None,
        None
    ]
    n_inputs = 2

    def call(self, img1, img2):
        return img1 ** 2 - img2 ** 2

gaussian_kernel_5 = (
    tf.constant(
        [
            [1, 4, 6, 4, 6],
            [4, 16, 24, 16, 4],
            [6, 24, 36, 24, 6],
            [4, 16, 24, 16, 4],
            [1, 4, 6, 4, 1],
        ],
        dtype=DEFAULT_DTYPE,
    )
    / 256,
)
gaussian_kernel_3 = (
    tf.constant([[1, 2, 1], [2, 4, 2], [1, 2, 1]], dtype=DEFAULT_DTYPE) / 16
)
sharpen_kernel_5 = (
    tf.constant(
        [
            [1, 4, 6, 4, 6],
            [4, 16, 24, 16, 4],
            [6, 24, -476, 24, 6],
            [4, 16, 24, 16, 4],
            [1, 4, 6, 4, 1],
        ],
        dtype=DEFAULT_DTYPE,
    )
    * -1
    / 256,
)
sharpen_kernel_3 = tf.constant(
    [[0, -1, 0], [-1, 5, -1], [0, -1, 0]], dtype=DEFAULT_DTYPE
)

gaussian_kernel_3 = tf.tile(tf.reshape(gaussian_kernel_3, (3, 3, 1, 1)), [1, 1, 3, 1])
gaussian_kernel_5 = tf.tile(tf.reshape(gaussian_kernel_5, (5, 5, 1, 1)), [1, 1, 3, 1])
sharpen_kernel_3 = tf.tile(tf.reshape(sharpen_kernel_3, (3, 3, 1, 1)), [1, 1, 3, 1])
sharpen_kernel_5 = tf.tile(tf.reshape(sharpen_kernel_5, (5, 5, 1, 1)), [1, 1, 3, 1])

class Convolve(BaseLeafLayer):

    n_inputs = 2

    shapes = [
        None,
        (None, None, None, 1)
    ]

    def __init__(
        self, width, height, stride=None, padding=None
    ):
        super().__init__(width, height)
        self.stride = stride or (1, 1, 1, 1)
        self.padding = padding or "SAME"

    def call(self, img, kernel):

        if img.shape[-1] == 1:
            img = tf.tile(img, [1] * (img.ndim - 1) + [3])

        return tf.nn.depthwise_conv2d(
            img, kernel, strides=self.stride, padding=self.padding
        )


class StaticConvolve(Convolve):
    shapes = [
        None,
        (None, None, None, 1)
    ]
    n_inputs = 2
    
class Gauss3x3Convolve(StaticConvolve):
    shapes = [
        None,
    ]
    n_inputs = 1

    def call(self, img):
        return super().call(img, gaussian_kernel_3)

class Gauss5x5Convolve(StaticConvolve):
    shapes = [
        None,
    ]
    n_inputs = 1

    def call(self, img):
        return super().call(img, gaussian_kernel_5)

class Sharpen3x3Convolve(StaticConvolve):
    shapes = [
        None,
    ]
    n_inputs = 1

    def call(self, img):
        return super().call(img, sharpen_kernel_3)


class Sharpen5x5Convolve(StaticConvolve):
    shapes = [
        None,
    ]
    n_inputs = 1

    def call(self, img):
        return super().call(img, sharpen_kernel_5)


BUILD_CLASSES = {
    "Color": Color,
    "RGBColor": RGBColor,
    "Cone": Cone,
    "Ellipse": Ellipse,
    "Circle": Circle,
    "SafeDivide": SafeDivide,
    "Multiply": Multiply,
    "Gradient": Gradient,
    "RGBGradient": RGBGradient,
    "Color": Color,
    "Noise": Noise,
    "Sum": Sum,
    "Sub": Sub,
    "Gauss3x3Convolve": Gauss3x3Convolve,
    "Gauss5x5Convolve": Gauss5x5Convolve,
    "Sharpen3x3Convolve": Sharpen3x3Convolve,
    "Sharpen5x5Convolve": Sharpen5x5Convolve,
    "Cosine": Cosine,
    "Sigmoid": Sigmoid,
}
LEAF_CLASSES = ["ConstantColor", "Noise", "Color", "Gradient", "RGBGradient"]
    

if __name__ == "__main__":
    import matplotlib.pyplot as plt
    for name, cls in BUILD_CLASSES.items():
        print(name, "\n" * 4)
        if len(cls.shapes) == 0:
            print(cls().call().shape)
        else:
            shapes = [
                (7, 100, 100, 3) if cls.shapes[i] is None else (7, *((x or 100) for x in cls.shapes[i][1:]))
                for i in range(len(cls.shapes))
            ]
            inputs = [
                tf.random.uniform(shape, 0, 1)
                for shape in shapes
            ]
            print("\n" * 4)
            img = cls(100, 100).call(*inputs)

            plt.figure()
            plt.imshow(img.numpy()[0])
            plt.title(name)
            plt.show()
            
