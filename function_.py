import os
import sys
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


def linear_mesh(width=None, height=None, channels=None, batch_dim=True):
    x = tf.linspace(-1.0, 1.0, width)
    y = tf.linspace(-1.0, 1.0, height)
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

    @classproperty
    def n_inputs(cls):
        return len(cls.input_shapes)

    def create_inputs(self):
        return [Input(shape=shape) for shape in self.input_shapes]


class ConstantColor(BaseLayer):
    input_shapes = ()

    def __init__(self, width, height, initializer=None, seed=42):
        super().__init__(width, height)
        initializer = initializer or get_initializer("PositiveBounded")
        self.value = self.add_weight(
            name="value", shape=(1, 1, 1, 3), initializer=initializer
        )

    def call(self, *inputs):
        return tf.tile(self.value, [1, self.width, self.height, 1])

class Constant(BaseLayer):
    input_shapes = ()

    def __init__(self, width, height, initializer=None, seed=42):
        super().__init__(width, height)
        initializer = initializer or get_initializer("PositiveBounded")
        self.value = self.add_weight(
            name="value", shape=(1, 1, 1, 1), initializer=initializer
        )

    def call(self, *inputs):
        return tf.tile(self.value, [1, self.width, self.height, 3])


class Cone(BaseLayer):

    input_shapes = ()

    def __init__(self, width, height, initializer=None, seed=42):
        super().__init__(width, height)
        initializer = initializer or (
            get_initializer("CentralizedPoint", seed=seed),
            get_initializer("Radius", seed=seed),
        )
        self.center = self.add_weight(
            name="center", shape=(2,), initializer=initializer[0]
        )
        self.radius = self.add_weight(
            name="radius", shape=(2,), initializer=initializer[1]
        )

    def call(self, *inputs):
        x, y = linear_mesh(self.width, self.height, batch_dim=True)

        ellipsoid = K.sqrt(
            ((x - self.center[0]) / self.radius[0]) ** 2
            + ((y - self.center[1]) / self.radius[1]) ** 2
        )
        return ellipsoid


class Ellipse(BaseLayer):

    input_shapes = ()

    def __init__(self, width, height, initializer=None, seed=42):
        super().__init__(width, height)
        initializer = initializer or (
            get_initializer("CentralizedPoint", seed=seed),
            get_initializer("Radius", seed=seed),
        )
        self.center = self.add_weight(
            name="center", shape=(2,), initializer=initializer[0]
        )
        self.radius = self.add_weight(
            name="radius", shape=(2,), initializer=initializer[1]
        )

    def call(self, *inputs):
        x, y = linear_mesh(self.width, self.height, batch_dim=True)

        ellipsoid = K.sqrt(
            ((x - self.center[0]) / self.radius[0]) ** 2
            + ((y - self.center[1]) / self.radius[1]) ** 2
        )
        ellipsoid = K.cast(ellipsoid >= 1, x.dtype)
        return ellipsoid


class Circle(BaseLayer):

    input_shapes = ()

    def __init__(self, width, height, initializer=None, seed=42):
        super().__init__(width, height)
        initializer = initializer or (
            get_initializer("CentralizedPoint"),
            get_initializer("Radius"),
        )
        self.center = self.add_weight(
            name="center", shape=(2,), initializer=initializer[0]
        )
        self.radius = self.add_weight(
            name="radius", shape=(1,), initializer=initializer[1]
        )

    def call(self, *inputs):
        x, y = linear_mesh(self.width, self.height, batch_dim=True)

        ellipsoid = K.sqrt(
            ((x - self.center[0]) / self.radius) ** 2
            + ((y - self.center[1]) / self.radius) ** 2
        )
        ellipsoid = K.cast(ellipsoid >= 1, x.dtype)
        return ellipsoid


class SafeDivide(BaseLayer):

    input_shapes = (
        (None, None, None),
        (None, None, None),
    )

    def __init__(self, width, height, eps=1e-3):
        super().__init__(width, height)
        self.eps = eps

    def call(self, x, y):

        return tf.clip_by_value(x / y, -1 / self.eps, 1 / self.eps)


class Multiply(BaseLayer):

    input_shapes = (
        (None, None, None),
        (None, None, None),
    )

    def call(self, x, y):
        return x * y


class Gradient(BaseLayer):

    input_shapes = ()

    def __init__(self, width=None, height=None):
        super().__init__(width, height)
        self.grad = self.add_weight(
            name="grad", shape=(2, 1, 1, 1, 1), initializer=get_initializer("Normal")
        )

    def call(self, *inputs):
        x, y = linear_mesh(self.width, self.height, batch_dim=True)
        return self.grad[0] * x + self.grad[1] * y


class RGBGradient(BaseLayer):

    input_shapes = ()

    def __init__(self, width=None, height=None):
        super().__init__(width, height)
        self.grad = self.add_weight(
            name="grad", shape=(2, 1, 1, 1, 3), initializer=get_initializer("Normal")
        )

    def call(self, *inputs):

        x, y = linear_mesh(self.width, self.height, channels=3, batch_dim=True)
        return self.grad[0] * x + self.grad[1] * y


class Color(BaseLayer):
    input_shapes = ()

    def __init__(self, width, height):
        super().__init__(width, height)
        self.color = self.add_weight(
            name="color",
            shape=(1, 1, 1, 3),
            initializer=get_initializer("PositiveBounded"),
        )

    def call(self, *inputs):
        return tf.tile(self.color, [1, self.width, self.height, 1])


class Noise(BaseLayer):

    input_shapes = ()

    def call(self, *inputs):
        return tf.random.uniform((1, self.width, self.height, 3), 0, 1)


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

        if img.shape[-1] == 1:
            img = tf.tile(img, [1] * (img.ndim - 1) + [3])

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


class LeafLayer(BaseLayer):
    def __init__(self, width, height):
        super().__init__(width, height)

    def call(self):
        return Input(shape=(self.width, self.height, 3))


BUILD_CLASSES = {
    # "Constant": Constant,
    "ConstantColor": ConstantColor,
    "Constant": Constant,
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
    # "StaticConvolve": StaticConvolve,
    # "Convolve": Convolve,
    "Gauss3x3Convolve": Gauss3x3Convolve,
    "Gauss5x5Convolve": Gauss5x5Convolve,
    "Sharpen3x3Convolve": Sharpen3x3Convolve,
    "Sharpen5x5Convolve": Sharpen5x5Convolve,
    # "Input": Input,
}
LEAF_CLASSES = ["ConstantColor", "Noise", "Color", "Gradient", "RGBGradient"]
    
if __name__ == "__main__":

    for name, cls in BUILD_CLASSES.items():
        print(f"Testing {name}")
        layer = cls(256, 256)
        inputs = list()
        for shape in cls.input_shapes:
            inputs.append(
                tf.random.uniform([1] + [(s if s is not None else 64) for s in shape])
            )
        print(layer(*inputs).shape)
