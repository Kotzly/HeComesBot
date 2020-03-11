import optparse
import json
from config import load_default_config, config_tuple
import time

def parse_sysargs(parser):
    options, _ = parser.parse_args()
    use_default = options.default
    input_config = {"background_path": options.background_path,
                    "dims": [int(n) for n in options.dims.split('x')],
                    "seed": int(options.seed),
                    "fontsize": int(options.fontsize)}
    config = load_default_config()._asdict()
    if not use_default:
        config.update(input_config)
        with open("./config.json", "w") as config_file:
            json.dump(config, config_file, indent=4)
    return config_tuple(**config)

def parse_cmd_args():
    parser = optparse.OptionParser()
    parser.add_option('-o', '--output', dest='background_path', default="./background.png", help='Write output to FILE', metavar='FILE')
    parser.add_option('-d', '--dims', dest='dims', default='512x512', help='Image width x height, e.g. 320x240')
    parser.add_option('-s', '--seed', dest='seed', default=int(1000 * time.time()), help='Random seed (uses system time by default)')
    parser.add_option('-f', '--fontsize', dest='fontsize', default=40, help='Fontsize for text in image.')
    parser.add_option('-D', '--default_config', dest='default', default=False, action="store_true", help='Use default CONFIGuration file.')
    config = parse_sysargs(parser)
    return config
