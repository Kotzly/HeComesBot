import numpy as np

from hecomes.artgen.func_utils import hsv_to_rgb

COLOR_SPACES = ("rgb", "hsv", "cmy")


def render_frame(raw_batch, color_space, dx, dy):
    """Convert a raw eval result (N, H, W, C) to a displayable (dy, dx, 3) uint8 array."""
    frame = np.broadcast_to(raw_batch[0], (dy, dx, raw_batch.shape[-1])).copy()
    if color_space == "hsv":
        hsv = np.stack(
            [
                frame[..., 0] % 1.0,
                frame[..., 1].clip(0, 1),
                frame[..., 2].clip(0, 1),
            ],
            axis=-1,
        )
        frame = hsv_to_rgb(hsv)
    elif color_space == "cmy":
        frame = 1.0 - frame.clip(0, 1)
    else:
        frame = frame.clip(0, 1)
    if frame.shape[-1] != 3:
        frame = frame[..., :3]
    return np.rint(frame * 255).astype(np.uint8)
