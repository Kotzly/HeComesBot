import random
from os.path import join

import markovify
import numpy as np
from PIL import Image, ImageDraw, ImageFont

from artgen.tree import get_random_function, log_tree_to_file


def random_sequence_length(max_length, min_length):
    if max_length is None and min_length is not None:
        sequence_length = min_length
    elif max_length is not None and min_length is None:
        sequence_length = max_length
    elif max_length is not None and min_length is not None:
        sequence_length = np.random.randint(min_length, max_length + 1)
    else:
        sequence_length = 1
    return sequence_length


def make_text(quotes_path, max_length=None, min_length=None, state_size=2, seed=42):
    with open(quotes_path, encoding="utf8") as quote_file:
        text = quote_file.read()

    text_model = markovify.NewlineText(text, state_size=state_size)
    sequence_length = random_sequence_length(max_length, min_length)

    random.seed(seed)
    return text_model.make_short_sentence(sequence_length)


def build_img(
    min_depth=5,
    max_depth=15,
    dx=100,
    dy=100,
    weights=None,
    log_filepath="tree.txt",
    seed=42,
):
    np.random.seed(seed % (2**32 - 1))

    def _build_img(depth=0):
        n_args, func = get_random_function(
            depth, p=weights, min_depth=min_depth, max_depth=max_depth
        )
        log_tree_to_file(func, depth, log_filepath=log_filepath)
        args = [_build_img(depth + 1) for i in range(n_args)]
        kwargs = dict(dx=dx, dy=dy) if n_args == 0 else {}
        try:
            return func(*args, **kwargs)
        except Exception as e:
            print(func.__name__, str(e))
            raise e

    return _build_img(depth=0)


def make_background(
    dx,
    dy,
    min_depth=5,
    max_depth=15,
    seed=42,
    save_filepath=None,
    log_path=None,
    personality=None,
):
    log_filepath = join(log_path, f"tree_{seed}.txt") if log_path is not None else None

    fail_counter = 0
    while True:
        img = build_img(
            min_depth=min_depth,
            max_depth=max_depth,
            dx=dx,
            dy=dy,
            weights=personality,
            log_filepath=log_filepath,
            seed=seed,
        )
        if img.shape == (dy, dx, 3):
            break
        else:
            fail_counter += 1
            if fail_counter > 10:
                raise Exception(
                    f"Too many failures when building image (got {img.shape})"
                )
            print(
                f"Failed build image (wrong dimensions). Trying again with seed {seed+1} [{fail_counter}/{10}]"
            )
            seed += 1

    img_8bit = np.rint(img.clip(0.0, 1.0) * 255.0).astype(np.uint8)

    if save_filepath is not None:
        Image.fromarray(img_8bit).save(save_filepath)
        print("Seed {}; wrote output to {}".format(seed, save_filepath))
    return img_8bit


def text_wrap(text, font, max_width):
    lines = []

    if font.getsize(text)[0] <= max_width:
        lines.append(text)
    else:
        words = text.split(" ")
        i = 0
        while i < len(words):
            line = ""
            while i < len(words) and font.getsize(line + words[i])[0] <= max_width:
                line = line + words[i] + " "
                i += 1
            if not line:
                line = words[i]
                i += 1
            lines.append(line)
    return lines


def combine_image(
    text,
    fontsize=40,
    background_path="./background.png",
    output_path="./output.png",
    font_path="./zalgo.ttf",
):
    img = Image.open(background_path)
    width, height = img.size
    drawing = ImageDraw.Draw(img)
    image_size = (width, height)
    textcolour = (0, 0, 0)
    text_outline = (255, 255, 255)
    fnt = ImageFont.truetype(font_path, fontsize, encoding="unic")

    draw = ImageDraw.Draw(img)
    outline_amount = 3

    line_height = fnt.getsize("hg")[1]
    x = random.randint(0, 50)
    y = random.randint(0, 150)

    for adj in range(outline_amount):
        draw.text((x - adj, y), text, font=fnt, fill=text_outline)
        draw.text((x + adj, y), text, font=fnt, fill=text_outline)
        draw.text((x, y + adj), text, font=fnt, fill=text_outline)
        draw.text((x, y - adj), text, font=fnt, fill=text_outline)
        draw.text((x - adj, y + adj), text, font=fnt, fill=text_outline)
        draw.text((x + adj, y + adj), text, font=fnt, fill=text_outline)
        draw.text((x - adj, y - adj), text, font=fnt, fill=text_outline)
        draw.text((x + adj, y - adj), text, font=fnt, fill=text_outline)

    if len(text) >= 55:
        wrapped_text = text_wrap(text, fnt, image_size[0])
        for lines in wrapped_text:
            if lines is None:
                break
            else:
                drawing.text((x, y), lines, fill=textcolour, font=fnt)
                y = y + line_height
    else:
        drawing.text((x, y), text, font=fnt, fill=textcolour)

    img.save(output_path, optimize=True)
    print("Text written")
