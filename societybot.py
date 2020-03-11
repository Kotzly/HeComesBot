import numpy as np, optparse, random, time, schedule
import markovify
import facebook
import json
from PIL import Image
import os
from os.path import join
import shutil
from config import get_config, load_default_config
from command import parse_cmd_args
from build import make_background, combine_image

def make_text(quotes_path, state_size=2, sequence_length=50, seed=42):
    # Get raw text as string.
    with open(quotes_path, encoding='utf8') as quote_file:
       text = quote_file.read()

    # Build the model.
    text_model = markovify.NewlineText(text, state_size=state_size)
    
    random.seed(seed)
    return text_model.make_short_sentence(sequence_length)

def post_to_facebook(post_text, filepath="./output.png", post=True, token=""):
    #obvs token is hidden
    if not post:
        return
    graph = facebook.GraphAPI(token)
    with open(filepath, 'rb') as image_file:
        post_id = graph.put_photo(image = image_file, message=post_text)['post_id']
    print(f"Success in uploading {post_id} to facebook")

def rename_with_seed(filename):
    filename = filename.split("/")[-1].split(".")[0]
    filename += "_" + str(CONFIG.seed)
    filename += ".png"
    return filename

def log_image():
    if CONFIG.history_path is not None:
        for filename in [CONFIG.output_path, CONFIG.background_path]:
            with Image.open(filename) as file:
                filename = rename_with_seed(filename)
                history_path = join(CONFIG.history_path, filename)
                file.save(history_path, optimize=True)


def log_title(seed, title):
    with open("./log.txt", "a") as log_file:
        timestamp = time.time()
        text = "{}\t{}\t{}\n".format(timestamp, seed, title)
        log_file.write(text)

def restart():
    if os.path.isdir(CONFIG.history_path):
        shutil.rmtree(CONFIG.history_path)
    if os.path.isfile("./log.txt"):
        os.remove("./log.txt")

def job():
    global CONFIG
    CONFIG = parse_cmd_args()
    if CONFIG.history_path is not None and not os.path.isdir(CONFIG.history_path):
        os.makedirs(CONFIG.history_path)

    title = None
    while title is None:
        sequence_length = np.random.randint(CONFIG.min_sequence_length, CONFIG.max_sequence_length + 1)
        title = make_text(CONFIG.quotes_path,
                          CONFIG.markov_model_state_size,
                          sequence_length,
                          CONFIG.seed)
    title = title.lower()
    make_background_kwargs = dict(dx=CONFIG.dims[0],
                                  dy=CONFIG.dims[1],
                                  min_depth=CONFIG.min_depth,
                                  max_depth=CONFIG.max_depth,
                                  seed=CONFIG.seed,
                                  save_filepath=CONFIG.background_path,
                                  log_path=CONFIG.tree_log_path,
                                  personality=CONFIG.personality)
    combine_kwargs = dict(background_path=CONFIG.background_path,
                          fontsize=CONFIG.fontsize,
                          output_path=CONFIG.output_path,
                          font_path=CONFIG.font_path)
    post_kwargs = dict(filepath=CONFIG.output_path,
                       post=CONFIG.post_to_facebook,
                       token=CONFIG.token)

    # try:
    make_background(**make_background_kwargs)
    # except:
    #     title = "failed"

    combine_image(title, **combine_kwargs)
    post_to_facebook(title, **post_kwargs)

    log_image()

    log_title(CONFIG.seed, title)

if __name__ == '__main__':

    if get_config("restart"):
        CONFIG = load_default_config()
        restart()

    delay = get_config("post_delay_seconds")
    schedule.every(delay).seconds.do(job).run()
    
    while True:
        schedule.run_pending()
        time.sleep(1)
