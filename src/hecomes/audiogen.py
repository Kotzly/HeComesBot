"""Tree-based audio generator.

Mirrors the visual tree algorithm in :mod:`hecomes.artgen.tree`, but every
node produces a 1D float32 waveform of shape ``(N,)`` instead of an
``(H, W, C)`` image.

Leaves are oscillators (sine, saw, square, triangle, noise). Unary ops are
distortions and phase shift. Binary ops each take exactly 2 scalar
parameters and combine two child waveforms.
"""

from __future__ import annotations

import wave
from dataclasses import dataclass, field
from typing import Callable

import numpy as np

SAMPLE_RATE = 44_100
TWO_PI = 2.0 * np.pi


# ── Leaves (arity 0) ─────────────────────────────────────────────────────────

def sine(n, freq=440.0, phase=0.0):
    t = np.arange(n, dtype=np.float32) / SAMPLE_RATE
    return np.sin(TWO_PI * freq * t + phase).astype(np.float32)


def saw(n, freq=110.0, phase=0.0):
    t = np.arange(n, dtype=np.float32) / SAMPLE_RATE
    x = (freq * t + phase / TWO_PI) % 1.0
    return (2.0 * x - 1.0).astype(np.float32)


def square(n, freq=220.0, duty=0.5):
    t = np.arange(n, dtype=np.float32) / SAMPLE_RATE
    x = (freq * t) % 1.0
    return np.where(x < duty, 1.0, -1.0).astype(np.float32)


def triangle(n, freq=330.0, phase=0.0):
    t = np.arange(n, dtype=np.float32) / SAMPLE_RATE
    x = (freq * t + phase / TWO_PI) % 1.0
    return (4.0 * np.abs(x - 0.5) - 1.0).astype(np.float32)


def noise(n, amp=0.5):
    return (amp * np.random.uniform(-1.0, 1.0, size=n)).astype(np.float32)


# ── Unary ops (arity 1) ──────────────────────────────────────────────────────

def soft_clip(x, drive=3.0):
    return np.tanh(drive * x).astype(np.float32)


def hard_clip(x, threshold=0.5):
    return np.clip(x, -threshold, threshold).astype(np.float32)


def wave_fold(x, gain=2.0):
    y = gain * x
    y = np.abs(np.mod(y + 1.0, 4.0) - 2.0) - 1.0
    return y.astype(np.float32)


def bit_crush(x, bits=4):
    levels = max(2, 2 ** int(bits))
    return (np.round(x * (levels / 2)) / (levels / 2)).astype(np.float32)


def phase_shift(x, delay_ms=5.0):
    shift = int(SAMPLE_RATE * delay_ms / 1000.0)
    return np.roll(x, shift).astype(np.float32)


# ── Binary ops (arity 2) ─────────────────────────────────────────────────────

def crossfade(a, b, mix=0.5, tilt=0.0):
    n = a.shape[0]
    ramp = np.linspace(-1.0, 1.0, n, dtype=np.float32)
    w = np.clip(mix + tilt * ramp, 0.0, 1.0)
    return ((1.0 - w) * a + w * b).astype(np.float32)


def ring_am(a, b, depth=0.8, offset=0.2):
    return (a * (offset + depth * b)).astype(np.float32)


def wave_fold_mod(a, b, threshold=0.6, gain=1.5):
    limit = threshold + gain * np.abs(b) * 0.3
    y = a / np.maximum(limit, 1e-3)
    y = np.abs(np.mod(y + 1.0, 4.0) - 2.0) - 1.0
    return (y * limit).astype(np.float32)


def fm_like(a, b, index=0.3, ratio=1.0):
    n = a.shape[0]
    idx = np.arange(n, dtype=np.float32) + index * ratio * b * SAMPLE_RATE * 0.001
    idx = np.clip(idx, 0, n - 1).astype(np.int32)
    return a[idx].astype(np.float32)


# ── Function registry ────────────────────────────────────────────────────────

@dataclass
class FunctionDef:
    func: Callable
    arity: int
    generate: Callable = field(default=lambda: {})


def _gen_sine():
    return {"freq": float(np.random.uniform(80, 1200)), "phase": float(np.random.uniform(0, TWO_PI))}


def _gen_saw():
    return {"freq": float(np.random.uniform(40, 600)), "phase": float(np.random.uniform(0, TWO_PI))}


def _gen_square():
    return {"freq": float(np.random.uniform(60, 800)), "duty": float(np.random.uniform(0.2, 0.8))}


def _gen_triangle():
    return {"freq": float(np.random.uniform(60, 800)), "phase": float(np.random.uniform(0, TWO_PI))}


def _gen_noise():
    return {"amp": float(np.random.uniform(0.1, 0.5))}


def _gen_soft_clip():
    return {"drive": float(np.random.uniform(1.5, 8.0))}


def _gen_hard_clip():
    return {"threshold": float(np.random.uniform(0.3, 0.9))}


def _gen_wave_fold():
    return {"gain": float(np.random.uniform(1.2, 3.0))}


def _gen_bit_crush():
    return {"bits": int(np.random.choice([2, 3, 4, 5, 6]))}


def _gen_phase_shift():
    return {"delay_ms": float(np.random.uniform(1.0, 25.0))}


def _gen_crossfade():
    return {"mix": float(np.random.uniform(0.2, 0.8)), "tilt": float(np.random.uniform(-0.4, 0.4))}


def _gen_ring_am():
    return {"depth": float(np.random.uniform(0.3, 1.0)), "offset": float(np.random.uniform(0.0, 0.5))}


def _gen_wave_fold_mod():
    return {"threshold": float(np.random.uniform(0.3, 0.9)), "gain": float(np.random.uniform(0.5, 2.5))}


def _gen_fm_like():
    return {"index": float(np.random.uniform(0.05, 0.6)), "ratio": float(np.random.uniform(0.5, 2.0))}


REGISTRY = [
    FunctionDef(sine, 0, _gen_sine),
    FunctionDef(saw, 0, _gen_saw),
    FunctionDef(square, 0, _gen_square),
    FunctionDef(triangle, 0, _gen_triangle),
    FunctionDef(noise, 0, _gen_noise),
    FunctionDef(soft_clip, 1, _gen_soft_clip),
    FunctionDef(hard_clip, 1, _gen_hard_clip),
    FunctionDef(wave_fold, 1, _gen_wave_fold),
    FunctionDef(bit_crush, 1, _gen_bit_crush),
    FunctionDef(phase_shift, 1, _gen_phase_shift),
    FunctionDef(crossfade, 2, _gen_crossfade),
    FunctionDef(ring_am, 2, _gen_ring_am),
    FunctionDef(wave_fold_mod, 2, _gen_wave_fold_mod),
    FunctionDef(fm_like, 2, _gen_fm_like),
]


# ── Tree ─────────────────────────────────────────────────────────────────────

@dataclass
class Node:
    func: FunctionDef
    params: dict
    children: list = field(default_factory=list)


def pick_function(depth, min_depth, max_depth):
    candidates = [
        fd for fd in REGISTRY
        if (fd.arity > 0 and depth < max_depth) or (fd.arity == 0 and depth >= min_depth)
    ]
    return candidates[np.random.randint(len(candidates))]


def build_tree(depth=0, min_depth=2, max_depth=5):
    fd = pick_function(depth, min_depth, max_depth)
    params = fd.generate()
    children = [build_tree(depth + 1, min_depth, max_depth) for _ in range(fd.arity)]
    return Node(func=fd, params=params, children=children)


def eval_tree(node, n_samples):
    if node.func.arity == 0:
        return node.func.func(n_samples, **node.params)
    args = [eval_tree(c, n_samples) for c in node.children]
    return node.func.func(*args, **node.params)


def describe(node, depth=0):
    pad = "  " * depth
    line = f"{pad}{node.func.func.__name__} {node.params}"
    lines = [line]
    for c in node.children:
        lines.extend(describe(c, depth + 1))
    return lines


# ── Output ───────────────────────────────────────────────────────────────────

def normalize(x, headroom=0.9):
    peak = float(np.max(np.abs(x))) or 1.0
    return (x * (headroom / peak)).astype(np.float32)


def write_wav(path, samples, sr=SAMPLE_RATE):
    pcm = np.clip(samples, -1.0, 1.0)
    pcm = (pcm * 32767.0).astype(np.int16)
    with wave.open(path, "wb") as f:
        f.setnchannels(1)
        f.setsampwidth(2)
        f.setframerate(sr)
        f.writeframes(pcm.tobytes())


def generate_wav(path, seconds, seed=None, min_depth=6, max_depth=10, verbose=False):
    """Build a random tree, evaluate it, and write a mono 16-bit WAV to ``path``.

    Returns the root node so callers can inspect it if they want.
    """
    if seed is not None:
        np.random.seed(seed % (2**32 - 1))
    n = int(seconds * SAMPLE_RATE)
    root = build_tree(min_depth=min_depth, max_depth=max_depth)
    if verbose:
        print("\n".join(describe(root)))
    y = normalize(eval_tree(root, n))
    write_wav(path, y)
    return root
