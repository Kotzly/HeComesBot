import json
from collections import namedtuple

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
