import numpy as np
import random
from function_ import BUILD_CLASSES
from PIL import Image, ImageDraw, ImageFont
from os.path import join, isfile
import config
import markovify
from tensorflow.keras.layers import Input
from tensorflow.keras.models import Model
import os

def random_sequence_length(max_length, min_length):
    if max_length is None and min_length is not None:
        sequence_length = min_length
    elif max_length is not None and min_length is not None:
        sequence_length = max_length
    elif max_length is not None and min_length is not None:
        sequence_length = np.random.randint(min_length, max_length + 1)
    else:
        sequence_length = 1
    return sequence_length

def make_text(quotes_path, max_length=None, min_length=None, state_size=2, seed=42):
    with open(quotes_path, encoding='utf8') as quote_file:
       text = quote_file.read()

    text_model = markovify.NewlineText(text, state_size=state_size)
    sequence_length = random_sequence_length(max_length, min_length)

    random.seed(seed)
    return text_model.make_short_sentence(sequence_length)


def log_tree_to_file(func, depth, log_filepath="tree.txt"):
    if log_filepath is not None:
        mode = "a" if isfile(log_filepath) else "w"
        with open(log_filepath, mode) as log_file:
            log_file.write("|\t"*depth + func.__name__ + "\n")

def get_random_function(depth=0, min_depth=5, max_depth=15, weights=None):

    if weights is None:
        p = np.ones(len(BUILD_CLASSES))
    else:
        p = [weights[c] for c in BUILD_CLASSES.keys()]

    cls_list, weights = list(), list()

    for cls, w in zip(BUILD_CLASSES.values(), p):
        n_args = cls.n_inputs
        if (n_args == 0 and depth >= min_depth) or (n_args > 0 and depth < max_depth):
            cls_list.append(cls)
            weights.append(w)

    weights = list(np.array(weights)/sum(weights))
    cls = np.random.choice(cls_list, p=weights)

    return cls

def build_img(min_depth=5, max_depth=15, dx=256, dy=256, weights=None, seed=42):
    
    gen = np.random.default_rng(seed)
    dummy_inputs = [Input(shape=(1,))]
    inputs = list()

    def _build_node(depth=0):
        
        options = list(BUILD_CLASSES.values())
        cls = get_random_function(depth, weights=weights, min_depth=min_depth, max_depth=max_depth)
        n_args = cls.n_inputs

        args = [_build_node(depth + 1) for i in range(n_args)] or dummy_inputs

        if args is dummy_inputs:
            dummy_inputs = []
        assert not any([arg is None for arg in args])
        output = cls(dx, dy)(*args)
        assert output.shape[1] == dx
        assert output.shape[2] == dy
        return output

    layers = _build_node(depth=0)
    model = Model(inputs=dummy_inputs, outputs=layers)
    return model

if __name__ == "__main__":
    for i in range(100):
        print(i)
        build_img(8, 12, dx=2560, dy=1080, seed=i, weights=None)
