from dataclasses import dataclass, field
from typing import Callable, Optional

import numpy as np

from hecomes.artgen.primitives import (
    _gen_circle,
    _gen_color_rotate,
    _gen_cone,
    _gen_gradient,
    _gen_kaleidoscope,
    _gen_rand_color,
    _gen_sphere,
    absolute_value,
    blur,
    circle,
    circular_mean,
    circular_mean_far,
    color_rotate,
    cone,
    hue_diff,
    hue_rotate,
    kaleidoscope,
    mirrored_sigmoid,
    rand_color,
    saddle,
    safe_divide,
    safe_modulus,
    sharpen,
    sigmoid,
    sphere,
    swap_phase_amplitude,
    x_var,
    y_var,
)

# ── Registry ──────────────────────────────────────────────────────────────────


@dataclass
class FunctionDef:
    func: Callable
    arity: int
    params: list = field(default_factory=list)
    generate: Optional[Callable] = None


FUNCTION_REGISTRY = [
    # ── Leaves (arity 0) ──────────────────────────────────────────────────────
    FunctionDef(
        rand_color,
        0,
        params=[{"name": "color", "type": "color", "label": "Color"}],
        generate=_gen_rand_color,
    ),
    FunctionDef(
        x_var,
        0,
        params=[
            {
                "name": "angle",
                "type": "float",
                "min": 0.0,
                "max": 6.2832,
                "label": "Angle",
            },
            {"name": "color", "type": "color", "label": "Color"},
        ],
        generate=_gen_gradient,
    ),
    FunctionDef(
        y_var,
        0,
        params=[
            {
                "name": "angle",
                "type": "float",
                "min": 0.0,
                "max": 6.2832,
                "label": "Angle",
            },
            {"name": "color", "type": "color", "label": "Color"},
        ],
        generate=_gen_gradient,
    ),
    FunctionDef(
        cone,
        0,
        params=[
            {
                "name": "cx",
                "type": "float",
                "min": -2.0,
                "max": 2.0,
                "label": "Center X",
            },
            {
                "name": "cy",
                "type": "float",
                "min": -2.0,
                "max": 2.0,
                "label": "Center Y",
            },
            {
                "name": "rx",
                "type": "float",
                "min": 0.01,
                "max": 1.0,
                "label": "Radius X",
            },
            {
                "name": "ry",
                "type": "float",
                "min": 0.01,
                "max": 1.0,
                "label": "Radius Y",
            },
            {"name": "color", "type": "color", "label": "Color"},
        ],
        generate=_gen_cone,
    ),
    FunctionDef(
        sphere,
        0,
        params=[
            {"name": "cx", "type": "float", "min": -2.0, "max": 2.0, "label": "Center X"},
            {"name": "cy", "type": "float", "min": -2.0, "max": 2.0, "label": "Center Y"},
            {"name": "rx", "type": "float", "min": 0.01, "max": 1.0, "label": "Radius X"},
            {"name": "ry", "type": "float", "min": 0.01, "max": 1.0, "label": "Radius Y"},
            {"name": "color", "type": "color", "label": "Color"},
        ],
        generate=_gen_sphere,
    ),
    FunctionDef(
        circle,
        0,
        params=[
            {
                "name": "cx",
                "type": "float",
                "min": -2.0,
                "max": 2.0,
                "label": "Center X",
            },
            {
                "name": "cy",
                "type": "float",
                "min": -2.0,
                "max": 2.0,
                "label": "Center Y",
            },
            {
                "name": "rx",
                "type": "float",
                "min": 0.01,
                "max": 1.0,
                "label": "Radius X",
            },
            {
                "name": "ry",
                "type": "float",
                "min": 0.01,
                "max": 1.0,
                "label": "Radius Y",
            },
            {"name": "color", "type": "color", "label": "Color"},
        ],
        generate=_gen_circle,
    ),
    # ── Unary (arity 1) ───────────────────────────────────────────────────────
    FunctionDef(np.sin, 1),
    FunctionDef(np.cos, 1),
    FunctionDef(sigmoid, 1),
    FunctionDef(mirrored_sigmoid, 1),
    FunctionDef(absolute_value, 1),
    FunctionDef(sharpen, 1),
    FunctionDef(blur, 1),
    FunctionDef(
        color_rotate,
        1,
        params=[{"name": "angles", "type": "angles", "label": "Euler angles ZYX"}],
        generate=_gen_color_rotate,
    ),
    FunctionDef(
        kaleidoscope,
        1,
        params=[
            {
                "name": "n",
                "type": "int",
                "choices": [3, 5, 6, 7, 8],
                "label": "Segments",
            },
            {
                "name": "phase",
                "type": "float",
                "min": 0.0,
                "max": 6.2832,
                "label": "Phase",
            },
        ],
        generate=_gen_kaleidoscope,
    ),
    # ── Binary (arity 2) ──────────────────────────────────────────────────────
    FunctionDef(np.add, 2),
    FunctionDef(np.subtract, 2),
    FunctionDef(np.multiply, 2),
    FunctionDef(safe_divide, 2),
    FunctionDef(safe_modulus, 2),
    FunctionDef(saddle, 2),
    FunctionDef(swap_phase_amplitude, 2),
    # Circular hue functions (intended for HSV H-channel / independent-channels mode)
    FunctionDef(circular_mean, 2),
    FunctionDef(circular_mean_far, 2),
    FunctionDef(hue_diff, 2),
    FunctionDef(hue_rotate, 2),
]


# ── Derived exports ───────────────────────────────────────────────────────────

BUILD_FUNCTIONS = sorted(
    [(fd.arity, fd.func) for fd in FUNCTION_REGISTRY],
    key=lambda x: x[1].__name__,
)

REGISTRY_BY_NAME = {fd.func.__name__: fd for fd in FUNCTION_REGISTRY}


def generate_params(func_name):
    fd = REGISTRY_BY_NAME.get(func_name)
    if fd and fd.generate:
        return fd.generate()
    return {}


FUNC_PARAMS = {fd.func.__name__: fd.params for fd in FUNCTION_REGISTRY if fd.params}
