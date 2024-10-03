import os

os.environ["TF_ENABLE_ONEDNN_OPTS"] = "0"

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

DEFAULT_DTYPE = tf.keras.backend.floatx()

class ClassPropertyDescriptor(object):

    def __init__(self, fget, fset=None):
        self.fget = fget
        self.fset = fset

    def __get__(self, obj, klass=None):
        if klass is None:
            klass = type(obj)
        return self.fget.__get__(obj, klass)()

    def __set__(self, obj, value):
        if not self.fset:
            raise AttributeError("can't set attribute")
        type_ = type(obj)
        return self.fset.__get__(obj, type_)(value)

    def setter(self, func):
        if not isinstance(func, (classmethod, staticmethod)):
            func = classmethod(func)
        self.fset = func
        return self

def classproperty(func):
    if not isinstance(func, (classmethod, staticmethod)):
        func = classmethod(func)

    return ClassPropertyDescriptor(func)


def get_initializer(name, seed=42):
    return dict(
        PositiveBounded=tf.keras.initializers.RandomUniform(0, 1, seed=seed),
        Bounded=tf.keras.initializers.RandomUniform(-1, 1, seed=seed),
        Normal=tf.keras.initializers.RandomNormal(0, 1, seed=seed),
        PositiveNormal=tf.keras.initializers.TruncatedNormal(0, 0.5, seed=seed),
    )[name]


def linear_mesh(width=None, height=None, channels=None, batch_dim=True):
    y = tf.linspace(-1.0, 1.0, height)
    x = tf.linspace(-1.0, 1.0, width)
    x, y = tf.cast(x, DEFAULT_DTYPE), tf.cast(x, DEFAULT_DTYPE)
    x, y = tf.meshgrid(x, y)
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

    @classproperty
    def n_inputs(cls):
        return len(cls.input_shapes)
    

    def create_inputs(self):
        return [Input(shape=shape) for shape in self.input_shapes]


# Create a keras model that takes no inputs and outputs a constant tensor with values [1, 2, 3]
class Constant(BaseLayer):
    def __init__(self, shape=None, initializer=None, seed=42):
        super().__init__()
        self.shape = shape or ()
        initializer = initializer or get_initializer("PositiveBounded")
        self.value = self.add_weight(
            name="value", shape=self.shape, initializer=initializer
        )

    def call(self, inputs=None):
        return self.value


class ConstantColor(BaseLayer):
    input_shapes = ((None,),)

    def __init__(self, width, height, initializer=None, seed=42):
        super().__init__(width, height)
        initializer = initializer or get_initializer("PositiveBounded")
        self.value = self.add_weight(
            name="value", shape=(1, self.width, self.height, 3), initializer=initializer
        )

    def call(self, inputs=None):
        return self.value


class Cone(BaseLayer):

    input_shapes = ((2,), (2,))

    def call(self, center, radii):
        x, y = linear_mesh(self.width, self.height, batch_dim=True)

        ellipsoid = K.sqrt(
            ((x - center[..., 0]) / radii[..., 0]) ** 2
            + ((y - center[..., 0]) / radii[..., 1]) ** 2
        )
        return ellipsoid


class Ellipse(BaseLayer):

    input_shapes = ((2,), (2,))

    def call(self, center, radii):
        x, y = linear_mesh(self.width, self.height, batch_dim=True)

        ellipsoid = K.sqrt(
            ((x - center[..., 0]) / radii[..., 0]) ** 2
            + ((y - center[..., 1]) / radii[..., 1]) ** 2
        )
        ellipsoid = K.cast(ellipsoid >= 1, x.dtype)
        return ellipsoid


class Circle(BaseLayer):

    input_shapes = (
        (2,),
        (1,),
    )

    def call(self, center, radius):
        x, y = linear_mesh(self.width, self.height, batch_dim=True)

        ellipsoid = K.sqrt(
            ((x - center[..., 0]) / radius) ** 2 + ((y - center[..., 1]) / radius) ** 2
        )
        ellipsoid = K.cast(ellipsoid >= 1, x.dtype)
        return ellipsoid


class SafeDivide(BaseLayer):

    input_shapes = (
        (None,),
        (None,),
    )

    def __init__(self, width, height, eps=1e-3):
        super().__init__(width, height)
        self.eps = eps

    def call(self, x, y):

        return tf.clip_by_value(x / y, -1 / self.eps, 1 / self.eps)


class Multiply(BaseLayer):

    input_shapes = (
        (None,),
        (None,),
    )

    def call(self, x, y):
        return x * y


class Gradient(BaseLayer):

    input_shapes = (
        (1, 1, 1),
        (1, 1, 1),
    )

    def call(self, a, b):
        x, y = linear_mesh(self.width, self.height, batch_dim=True)
        return a * x + b * y


class RGBGradient(BaseLayer):

    input_shapes = (
        (1, 1, 3),
        (1, 1, 3),
    )

    def call(self, a, b):

        x, y = linear_mesh(self.width, self.height, channels=3, batch_dim=True)
        return a * x + b * y


class Color(BaseLayer):
    input_shapes = ((1, 1, 3),)

    def call(self, color):
        return color


class Noise(BaseLayer):

    input_shapes = ((None,),)

    def call(self, *inputs):
        return tf.random.uniform((1, 100, 100, 3), 0, 1)


class Sum(BaseLayer):

    input_shapes = (
        (None, None, None),
        (None, None, None),
    )

    def call(self, a, b):
        return a + b


class Sub(BaseLayer):

    input_shapes = (
        (None, None, None),
        (None, None, None),
    )

    def call(self, a, b):
        return a - b


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


class StaticConvolve(BaseLayer):

    input_shapes = ((None, None, 3),)

    def __init__(
        self, width, height, kernel=gaussian_kernel_3, stride=None, padding=None
    ):
        super().__init__(width, height)
        self.kernel = kernel
        self.stride = stride or (1, 1, 1, 1)
        self.padding = padding or "SAME"

    def call(self, img):
        return tf.nn.depthwise_conv2d(
            img, self.kernel, strides=self.stride, padding="SAME"
        )


# class Convolve(BaseLayer):

#     input_shapes = (
#         (None, None, 3),
#         (None, None, 1),
#     )

#     def __init__(self, width, height, stride=None, padding=None):
#         super().__init__(width, height)
#         self.stride = stride or (1, 1, 1, 1)
#         self.padding = padding or "SAME"

#     def call(self, img, kernel):
#         return tf.nn.depthwise_conv2d(img, kernel, strides=self.stride, padding="SAME")


class Gauss3x3Convolve(StaticConvolve):

    input_shapes = ((None, None, 3),)

    def __init__(self, width, height):
        super().__init__(width, height, gaussian_kernel_3)


class Gauss5x5Convolve(StaticConvolve):

    input_shapes = ((None, None, 3),)

    def __init__(self, width, height):
        super().__init__(width, height, gaussian_kernel_5)


class Sharpen3x3Convolve(StaticConvolve):

    input_shapes = ((None, None, 3),)

    def __init__(self, width, height):
        super().__init__(width, height, sharpen_kernel_3)


class Sharpen5x5Convolve(StaticConvolve):

    input_shapes = ((None, None, 3),)

    def __init__(self, width, height):
        super().__init__(width, height, sharpen_kernel_5)


BUILD_CLASSES = {
    # "Constant": Constant,
    # "ConstantColor": ConstantColor,
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
    "StaticConvolve": StaticConvolve,
    # "Convolve": Convolve,
    "Gauss3x3Convolve": Gauss3x3Convolve,
    "Gauss5x5Convolve": Gauss5x5Convolve,
    "Sharpen3x3Convolve": Sharpen3x3Convolve,
    "Sharpen5x5Convolve": Sharpen5x5Convolve,
}

if __name__ == "__main__":

    for name, cls in BUILD_CLASSES.items():
        print(f"Testing {name}")
        layer = cls(256, 256)
        inputs = list()
        for shape in cls.input_shapes:
            inputs.append(
                tf.random.uniform(
                    [1] + [(s if s is not None else 64) for s in shape]
                )
            )
        print(layer(*inputs).shape)
