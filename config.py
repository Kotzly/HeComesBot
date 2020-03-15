import json
from collections import namedtuple

DEFAULT_PERSONALITY_DICT = {"circle": 0,
                            "cone": 0,
                            "rand_color": 1,
                            "x_var": 1,
                            "y_var": 1,
                            "blur": 0,
                            "cos": 1,
                            "sharpen": 0,
                            "sigmoid": 0,
                            "sin": 1,
                            "add": 1,
                            "multiply": 1,
                            "saddle": 0,
                            "safe_divide": 1,
                            "subtract": 1,
                            "swap_phase_amplitude": 0}

def _load_default_config():
    with open("./config.json", "r") as config_file:
        config = json.load(config_file)
    return config

config_tuple = namedtuple("config", [*_load_default_config().keys()])

def load_default_config():
    with open("./config.json", "r") as config_file:
        config = json.load(config_file)
    return config_tuple(**config)

def get_config(key):
    config = load_default_config()._asdict()
    return config[key]

def load_personality_dict(path):
    if path is None:
        return DEFAULT_PERSONALITY_DICT
    with open(path, "r") as personality_file:
        personality = json.load(personality_file)
    personality_dict = {name:level[name] for level in personality.values() for name in level}
    return personality_dict

def load_personality_list(path):
    personality_dict = load_personality_dict(path)
    personality_list = [(name, personality_dict[name]) for name in personality_dict]
    personality_list = [weight for name, weight in sorted(personality_list, key=lambda x : x[0])]
    return personality_list

