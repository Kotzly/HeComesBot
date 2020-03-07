import numpy as np
import random
from functions import BUILD_FUNCTIONS
from PIL import Image, ImageDraw, ImageFont
from os.path import join, isfile

def log_tree_to_file(func, depth, log_filepath="tree.txt"):
    if log_filepath is not None:
        mode = "a" if isfile(log_filepath) else "w"
        with open(log_filepath, mode) as log_file:
            log_file.write("|\t"*depth + func.__name__ + "\n")

# Generate x and y images, with 3D shape so operations will correctly broadcast.
def get_random_function(depth=0, min_depth=5, max_depth=15, p=None):

    if p is None:
        p = np.ones(len(BUILD_FUNCTIONS))

    funcs, weights = list(), list()
    for (n_args, function), w in zip(BUILD_FUNCTIONS, p):
        if (n_args > 0 and depth < max_depth) or (n_args == 0 and depth >= min_depth):
            funcs.append((n_args, function))
            weights.append(w)
    weights = list(np.array(weights)/sum(weights))

    idx = np.random.choice(range(len(funcs)), p=weights)
    n_args, func = funcs[idx]
    return n_args, func

def build_img(depth=0, weights=None, log_filepath="tree.txt", seed=42):
    def _build_img(depth=0):
        n_args, func = get_random_function(depth, p=weights)
        log_tree_to_file(func, depth, log_filepath=log_filepath)
        args = [_build_img(depth + 1) for i in range(n_args)]
        return func(*args)
    return _build_img()


def make_background(dx, dy, min_depth=5, max_depth=15, seed=42, save_filepath=None, log_path=None, personality=None):
    # Recursively build an image using a random function.  Functions are built as a parse tree top-down,
    # with each node chosen randomly from the following list.  The first element in each tuple is the
    # number of required recursive calls and the second element is the function to evaluate the result.
    random.seed(seed)
    log_filepath = join(log_path, f"tree_{seed}.txt")
    img = build_img(weights=personality, log_filepath=log_filepath, seed=seed)
    # Ensure it has the right dimensions
    img = np.tile(img, (dx // img.shape[0], dy // img.shape[1], 3 // img.shape[2]))

    # Convert to 8-bit, send to PIL and save
    img_8bit = np.uint8(np.rint(img.clip(0.0, 1.0) * 255.0))
    
    if save_filepath is not None:
        Image.fromarray(img_8bit).save(save_filepath)
        print('Seed {}; wrote output to {}'.format(seed, save_filepath))
    return img_8bit


# This is being worked on, as wrapped text won't draw an outline for some reason
# as such, the max text size is 50 chars so it does not have to wrap
def text_wrap(text, font, max_width):

    lines = []

    #if the width is smaller than the image width, add to lines array
    if font.getsize(text)[0] <= max_width:
        lines.append(text)
    else:
        #split text by space to find words
        words = text.split(' ')

        #add each word to a line while the line is shorter than the image
        i = 0
        while i < len(words):
            line = ''
            while i < len(words) and font.getsize(line + words[i])[0] <= max_width:
                line = line + words[i] + " "
                i += 1
            if not line:
                line = words[i]
                i += 1
            #when the line is longer than the max width, add line to array
            lines.append(line)
        return lines


def combine_image(text, background_path="./background.png", output_path="./output.png", font_path="./zalgo.ttf"):

    #make the new image
    img = Image.open(background_path)
    width, height = img.size
    #setup d for drawing
    drawing = ImageDraw.Draw(img)
    #set image size
    image_size = (width, height)
    #set the colour of text & outline
    textcolour = (0, 0, 0)
    text_outline = (255, 255, 255)
    #create the font and set text size
    fnt = ImageFont.truetype(font_path, 40, encoding="unic")

    draw = ImageDraw.Draw(img)
    #Set outline amount, sets how many times we loop
    outline_amount = 3
    

    #set the height of each line
    line_height = fnt.getsize('hg')[1]
    x = random.randint(0, 50)
    y = random.randint(0, 150)

    # Make outline by re-drawing the text in every direction
    for adj in range(outline_amount):
        #move right
        draw.text((x-adj, y), text, font=fnt, fill=text_outline)
        #move left
        draw.text((x+adj, y), text, font=fnt, fill=text_outline)
        #move up
        draw.text((x, y+adj), text, font=fnt, fill=text_outline)
        #move down
        draw.text((x, y-adj), text, font=fnt, fill=text_outline)
        #diagnal left up
        draw.text((x-adj, y+adj), text, font=fnt, fill=text_outline)
        #diagnal right up
        draw.text((x+adj, y+adj), text, font=fnt, fill=text_outline)
        #diagnal left down
        draw.text((x-adj, y-adj), text, font=fnt, fill=text_outline)
        #diagnal right down
        draw.text((x+adj, y-adj), text, font=fnt, fill=text_outline)

    #check text's length to see if we need to wrap
    if len(text) >= 55:

        wrapped_text = text_wrap(text, fnt, image_size[0])

        for lines in wrapped_text:

            if lines == None:
                break
            else:
                #draw the lines on the image
                drawing.text((x,y), lines, fill=textcolour, font=fnt)
                y = y + line_height
    
    else:
        drawing.text((x,y), text, font=fnt, fill=textcolour)

    img.save(output_path, optimize=True)
    print('Text written')
