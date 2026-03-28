import os


def make_kwargs(config):
    make_text_kwargs = dict(
        quotes_path=config.quotes_path,
        min_length=config.min_sequence_length,
        max_length=config.max_sequence_length,
        state_size=config.markov_model_state_size,
        seed=config.seed,
    )

    make_background_kwargs = dict(
        dx=config.dims[0],
        dy=config.dims[1],
        min_depth=config.min_depth,
        max_depth=config.max_depth,
        seed=config.seed,
        save_filepath=config.background_path,
        log_path=config.tree_log_path,
        personality=config.personality,
    )

    combine_kwargs = dict(
        background_path=config.background_path,
        fontsize=config.fontsize,
        output_path=config.output_path,
        font_path=config.font_path,
    )

    return make_text_kwargs, make_background_kwargs, combine_kwargs


def hsv_to_rgb(hsv):
    """Convert (..., 3) HSV array in [0, 1] to RGB."""
    h, s, v = hsv[..., 0], hsv[..., 1], hsv[..., 2]
    i = (h * 6).astype(np.int32)
    f = h * 6 - i
    p = v * (1 - s)
    q = v * (1 - f * s)
    t = v * (1 - (1 - f) * s)
    i6 = i % 6
    r = np.select(
        [i6 == 0, i6 == 1, i6 == 2, i6 == 3, i6 == 4, i6 == 5], [v, q, p, p, t, v]
    )
    g = np.select(
        [i6 == 0, i6 == 1, i6 == 2, i6 == 3, i6 == 4, i6 == 5], [t, v, v, q, p, p]
    )
    b = np.select(
        [i6 == 0, i6 == 1, i6 == 2, i6 == 3, i6 == 4, i6 == 5], [p, p, t, v, v, q]
    )
    return np.stack([r, g, b], axis=-1).astype(np.float32)


def makedirs(paths):
    for path in paths:
        if not os.path.isdir(path):
            os.makedirs(path)
