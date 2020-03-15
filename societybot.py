import numpy as np, optparse, random, time, schedule
import facebook
import json
from PIL import Image
import os
from os.path import join
import shutil
from config import get_config, load_default_config
from command import parse_cmd_args
from build import make_background, combine_image, make_text
from utils import make_kwargs, makedirs

def post_to_facebook(post_text, filepath="./output.png", post=True, token=""):
    if not post:
        return
    graph = facebook.GraphAPI(token)
    with open(filepath, 'rb') as image_file:
        post_id = graph.put_photo(image = image_file, message=post_text)['post_id']
    print(f"Success in uploading {post_id} to facebook")

def rename_with_seed(filename, seed):
    filename = filename.split("/")[-1].split(".")[0]
    filename += "_" + str(seed) + ".png"
    return filename

def log_images(history_path, output_filepath, background_filepath, seed):
    if history_path is not None:
        filenames = [output_filepath, background_filepath]
        paths = ["outputs", "backgrounds"]
        filepaths = [join(history_path, path) for path in paths]
        if history_path is not None:
            for filename, filepath in zip(filenames, filepaths):
                with Image.open(filename) as file:
                    filename = rename_with_seed(filename, seed)
                    history_filepath = join(filepath, filename)
                    file.save(history_filepath, optimize=True)

def log_title(title, seed):
    with open("./log.txt", "a") as log_file:
        timestamp = time.time()
        text = "{}\t{}\t{}\n".format(timestamp, seed, title)
        log_file.write(text)

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
        paths = [CONFIG.history_path,
                 join(CONFIG.history_path, "outputs"),
                 join(CONFIG.history_path, "backgrounds")]
        makedirs(paths)
    make_text_kwargs, make_background_kwargs, combine_kwargs, post_kwargs = \
       make_kwargs(CONFIG)

    title = None
    while title is None:
        sequence_length = np.random.randint(CONFIG.min_sequence_length, CONFIG.max_sequence_length + 1)
        title = make_text(**make_text_kwargs)

    make_background(**make_background_kwargs)
    combine_image(title, **combine_kwargs)
    post_to_facebook(title, **post_kwargs)

    log_images(CONFIG.history_path, CONFIG.output_path, CONFIG.background_path, CONFIG.seed)
    log_title(title, CONFIG.seed)

if __name__ == '__main__':

    if get_config("restart"):
        history_path = get_config("history_path")
        restart(history_path)

    delay = get_config("post_delay_seconds")
    schedule.every(delay).seconds.do(job).run()
    
    while True:
        schedule.run_pending()
        time.sleep(1)
