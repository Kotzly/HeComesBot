import os

def make_kwargs(config):
    make_text_kwargs = dict(quotes_path=config.quotes_path,
                            min_length=config.min_sequence_length,
                            max_length=config.max_sequence_length,
                            state_size=config.markov_model_state_size,
                            seed=config.seed)

    make_background_kwargs = dict(dx=config.dims[0],
                                  dy=config.dims[1],
                                  min_depth=config.min_depth,
                                  max_depth=config.max_depth,
                                  seed=config.seed,
                                  save_filepath=config.background_path,
                                  log_path=config.tree_log_path,
                                  personality=config.personality)

    combine_kwargs = dict(background_path=config.background_path,
                          fontsize=config.fontsize,
                          output_path=config.output_path,
                          font_path=config.font_path)

    post_kwargs = dict(filepath=config.output_path,
                       post=config.post_to_facebook,
                       token=config.token)

    return make_text_kwargs, make_background_kwargs, combine_kwargs, post_kwargs

def makedirs(paths):
    for path in paths:
        if not os.path.isdir(path):
            os.makedirs(path)
