"""Per-frame visual features used to drive video-synchronised audio.

Frames are block-averaged down to roughly ``target x target`` pixels before
analysis, so the cost does not depend on the output resolution.

Features (one float per frame):

- ``motion``      mean absolute RGB change from the previous frame, in [0, 1]
- ``brightness``  mean luminance, in [0, 1]
- ``hue``         chroma-weighted circular mean hue, in [0, 1)
- ``saturation``  mean HSV saturation, in [0, 1]
- ``pan``         horizontal centroid of the motion, in [-1, 1] (0 when static)
"""

import numpy as np
from PIL import Image

from hecomes.artgen.func_utils import rgb_to_hsv

FEATURE_NAMES = ("motion", "brightness", "hue", "saturation", "pan")

_LUMA = np.array([0.299, 0.587, 0.114], dtype=np.float32)


def downsample(frames, target=128):
    """Block-average ``(n, H, W, C)`` uint8 frames to about ``target x target``.

    Uses ``by = H // target`` by ``bx = W // target`` pixel blocks (1080x1920
    becomes 135x128). Returns float32 RGB in [0, 1]; any alpha channel is
    dropped. Pillow's ``reduce`` is ~20x faster than a NumPy block mean here.
    """
    h, w = frames.shape[1:3]
    factor = (max(1, w // target), max(1, h // target))
    small = [np.asarray(Image.fromarray(f[..., :3]).reduce(factor)) for f in frames]
    return np.stack(small).astype(np.float32) / 255.0


class FrameFeatureExtractor:
    """Accumulate :data:`FEATURE_NAMES` from consecutive chunks of frames.

    Call it with each chunk, in display order (it is designed to be passed as
    ``on_chunk`` to :func:`hecomes.cli._video_utils.run_ffmpeg_pipeline`), then
    read :meth:`features`.
    """

    def __init__(self, target=128):
        self.target = target
        self._prev = None
        self._chunks = []

    def __call__(self, frames):
        small = downsample(frames, self.target)
        prev = small[:1] if self._prev is None else self._prev[None]
        self._prev = small[-1]

        diff = np.abs(np.diff(np.concatenate([prev, small]), axis=0)).mean(axis=-1)
        motion = diff.mean(axis=(1, 2))

        xs = np.linspace(-1.0, 1.0, small.shape[2], dtype=np.float32)
        column_motion = diff.sum(axis=1)
        pan = np.where(
            motion > 1e-6,
            (column_motion * xs).sum(axis=1) / np.maximum(column_motion.sum(axis=1), 1e-12),
            0.0,
        )

        with np.errstate(divide="ignore", invalid="ignore"):  # black pixels: 0/0 saturation
            hsv = rgb_to_hsv(small)
        chroma = hsv[..., 1] * hsv[..., 2]
        angle = 2.0 * np.pi * hsv[..., 0]
        hue = np.arctan2(
            (chroma * np.sin(angle)).sum(axis=(1, 2)),
            (chroma * np.cos(angle)).sum(axis=(1, 2)),
        ) / (2.0 * np.pi) % 1.0

        self._chunks.append(np.stack([
            motion,
            (small @ _LUMA).mean(axis=(1, 2)),
            hue,
            hsv[..., 1].mean(axis=(1, 2)),
            pan,
        ], axis=1).astype(np.float32))

    def features(self):
        """Return ``{name: (n_frames,) float32 array}`` for everything seen so far."""
        table = (
            np.concatenate(self._chunks)
            if self._chunks else np.zeros((0, len(FEATURE_NAMES)), dtype=np.float32)
        )
        return {name: table[:, i] for i, name in enumerate(FEATURE_NAMES)}
