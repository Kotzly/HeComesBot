import os
import shutil
from os.path import join

from PIL import Image

from hecomes.bot.image import combine_image, make_background, make_text
from hecomes.bot.command import parse_cmd_args
from hecomes.bot.utils import make_kwargs, makedirs
from hecomes.config import get_config


def rename_with_seed(filename, seed):
    filename = filename.split("/")[-1].split(".")[0]
    filename += "_" + str(seed) + ".png"
    return filename


def log_images(history_path, output_filepath, background_filepath, seed):
    if history_path is not None:
        filenames = [output_filepath, background_filepath]
        paths = ["outputs", "backgrounds"]
        filepaths = [join(history_path, path) for path in paths]
        for filename, filepath in zip(filenames, filepaths):
            with Image.open(filename) as file:
                filename = rename_with_seed(filename, seed)
                history_filepath = join(filepath, filename)
                file.save(history_filepath, optimize=True)


def restart(history_path):
    if history_path is not None:
        if os.path.isdir(history_path):
            shutil.rmtree(history_path)
    if os.path.isfile("./log.txt"):
        os.remove("./log.txt")


def job():
    global CONFIG
    CONFIG = parse_cmd_args()
    if CONFIG.history_path is not None:
        paths = [
            CONFIG.history_path,
            join(CONFIG.history_path, "outputs"),
            join(CONFIG.history_path, "backgrounds"),
        ]
        makedirs(paths)
    make_text_kwargs, make_background_kwargs, combine_kwargs = make_kwargs(CONFIG)

    title = None
    while title is None:
        title = make_text(**make_text_kwargs)

    make_background(**make_background_kwargs)
    combine_image(title, **combine_kwargs)
    log_images(
        CONFIG.history_path, CONFIG.output_path, CONFIG.background_path, CONFIG.seed
    )


if __name__ == "__main__":
    if get_config("restart"):
        history_path = get_config("history_path")
        restart(history_path)
    job()
