from os.path import isfile

import numpy as np

from hecomes.artgen.functions import BUILD_FUNCTIONS


def random_delta(alpha=5e-3):
    return np.random.choice([1, -1]) * alpha


def log_tree_to_file(func, depth, log_filepath="tree.txt"):
    if log_filepath is not None:
        mode = "a" if isfile(log_filepath) else "w"
        with open(log_filepath, mode) as log_file:
            log_file.write("|\t" * depth + func.__name__ + "\n")


def get_random_function(depth=0, min_depth=5, max_depth=15, p=None):
    if p is None:
        p = np.ones(len(BUILD_FUNCTIONS))

    funcs, weights = list(), list()

    for (n_args, function), w in zip(BUILD_FUNCTIONS, p):
        if (n_args > 0 and depth < max_depth) or (n_args == 0 and depth >= min_depth):
            funcs.append((n_args, function))
            weights.append(w)

    weights = list(np.array(weights) / sum(weights))
    idx = np.random.choice(range(len(funcs)), p=weights)
    n_args, func = funcs[idx]

    return n_args, func
