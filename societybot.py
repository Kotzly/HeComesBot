import numpy as np, optparse, random, time, schedule
from PIL import Image, ImageDraw, ImageFont
import markovify
import facebook
import json
import os
from os.path import join
from collections import namedtuple
import shutil

def load_default_config_():
    with open("./config.json", "r") as config_file:
        config = json.load(config_file)
    return config

config_tuple = namedtuple("config", [*load_default_config_().keys()])

def load_default_config():
    with open("./config.json", "r") as config_file:
        config = json.load(config_file)
    return config_tuple(**config)

def get_config(key):
    config = load_default_config()._asdict()
    return config[key]

def make_text(quotes_path, state_size=2, sequence_length=50):
    # Get raw text as string.
    with open(quotes_path, encoding='utf8') as quote_file:
       text = quote_file.read()

    # Build the model.
    text_model = markovify.NewlineText(text, state_size=state_size)
    
    return text_model.make_short_sentence(sequence_length)

def parse_sysargs(parser):
    options, _ = parser.parse_args()
    use_default = options.default
    input_config = {"background_path": options.background_path,
                    "dims": [int(n) for n in options.dims.split('x')],
                    "seed": int(options.seed)}
    config = load_default_config_()
    if not use_default:
        config.update(input_config)
    with open("./config.json", "w") as config_file:
        json.dump(config, config_file, indent=4)
    return config_tuple(**config)

def make_background():
     # Parse command-line options
    
    dX, dY = CONFIG.dims

    random.seed(CONFIG.seed)

    # Generate x and y images, with 3D shape so operations will correctly broadcast.
    xArray = np.linspace(0., 1., dX).reshape(1, -1, 1)
    yArray = np.linspace(0., 1., dY).reshape(-1, 1, 1)

    # Adaptor functions for the recursive generator
    # Note: using python's random module because numpy's doesn't handle seeds longer than 32 bits.
    def randColor():
        return np.random.rand(1, 1, 3)
    def x_var():
        return xArray
    def y_var():
        return yArray
    def circle():
        y = np.repeat(np.linspace(-1, 1, dY).reshape(-1, 1), dX, axis=1)/2
        x = np.repeat(np.linspace(-1, 1, dX).reshape(1, -1), dY, axis=0)/2
        r = 1 - np.random.rand()**2
        cx, cy = np.random.rand(2)
        mask = np.sqrt((x - cx)**2 + (y - cy)**2) > r
        circ = np.ones((dY, dX, 3))*randColor()
        circ[mask] = 0
        return circ

    def safe_divide(a, b, eps=1e-3):
        b[np.abs(b) < eps] = np.sign(b[np.abs(b) < eps])*eps
        b[b==0] = eps
        return np.divide(a, b)
    def sigmoid(x):
        return 1 / (1 + np.exp(-x))
    def mirrored_sigmoid(x):
        return 1 / (1 + np.exp(x))

    def get_random_function(depth=0, p=None):
        functions = (
                (0, randColor),
                (0, x_var),
                (0, y_var),
                (0, circle),

                (1, np.sin),
                (1, np.cos),
                (1, sigmoid),
                (1, mirrored_sigmoid),
                
                (2, np.add),
                (2, np.subtract),
                (2, np.multiply),
                (2, safe_divide), # 11 functions
            )
        if p is None:
            p = np.ones(len(functions))

        funcs, weights = list(), list()
        for (n_args, function), w in zip(functions, p):
            if (n_args > 0 and depth < CONFIG.max_depth) or (n_args == 0 and depth >= CONFIG.min_depth):
                funcs.append((n_args, function))
                weights.append(w)
        weights = list(np.array(weights)/sum(weights))

        idx = np.random.choice(range(len(funcs)), p=weights)
        n_args, func = funcs[idx]
        return n_args, func

    # Recursively build an image using a random function.  Functions are built as a parse tree top-down,
    # with each node chosen randomly from the following list.  The first element in each tuple is the
    # number of required recursive calls and the second element is the function to evaluate the result.
    def build_img(depth=0):
        n_args, func = get_random_function(depth, p=CONFIG.personality)
        tree_file = CONFIG.history_path + f"{CONFIG.seed}_tree.txt"
        mode = "a" if os.path.isfile(tree_file) else "w"
        with open(tree_file, mode) as file:
            file.write("|\t"*depth + func.__name__ + "\n")
            # file.write("\t"*(depth - 1) + ("\t" + "-"*6)*(depth > 0) + func.__name__ + "\n")

        args = list()
        for n in range(n_args):
            arg = build_img(depth + 1)
            args.append(arg)

        return func(*args)


    img = build_img()

    # Ensure it has the right dimensions
    img = np.tile(img, (dX // img.shape[0], dY // img.shape[1], 3 // img.shape[2]))

    # Convert to 8-bit, send to PIL and save
    img_8bit = np.uint8(np.rint(img.clip(0.0, 1.0) * 255.0))
    
    Image.fromarray(img_8bit).save(CONFIG.background_path)
    print('Seed {}; wrote output to {}'.format(CONFIG.seed, CONFIG.background_path))


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
    textOutline = (255, 255, 255)
    #create the font and set text size
    fnt = ImageFont.truetype(font_path, 40, encoding="unic")

    draw = ImageDraw.Draw(img)
    #Set outline amount, sets how many times we loop
    outlineAmount = 3
    

    #set the height of each line
    line_height = fnt.getsize('hg')[1]
    x = random.randint(0, 50)
    y = random.randint(0, 150)

    # Make outline by re-drawing the text in every direction
    for adj in range(outlineAmount):
        #move right
        draw.text((x-adj, y), text, font=fnt, fill=textOutline)
        #move left
        draw.text((x+adj, y), text, font=fnt, fill=textOutline)
        #move up
        draw.text((x, y+adj), text, font=fnt, fill=textOutline)
        #move down
        draw.text((x, y-adj), text, font=fnt, fill=textOutline)
        #diagnal left up
        draw.text((x-adj, y+adj), text, font=fnt, fill=textOutline)
        #diagnal right up
        draw.text((x+adj, y+adj), text, font=fnt, fill=textOutline)
        #diagnal left down
        draw.text((x-adj, y-adj), text, font=fnt, fill=textOutline)
        #diagnal right down
        draw.text((x+adj, y-adj), text, font=fnt, fill=textOutline)

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


def post_to_facebook(post_text, filepath="./output.png", post=True, token=""):
    #obvs token is hidden
    if not post:
        return
    graph = facebook.GraphAPI(token)
    with open(filepath, 'rb') as image_file:
        post_id = graph.put_photo(image = image_file, message=post_text)['post_id']
    print(f"Success in uploading {post_id} to facebook")


def parse_cmd_args():
    parser = optparse.OptionParser()
    parser.add_option('-o', '--output', dest='background_path', default="./background.png", help='Write output to FILE', metavar='FILE')
    parser.add_option('-d', '--dims', dest='dims', default='512x512', help='Image width x height, e.g. 320x240')
    parser.add_option('-s', '--seed', dest='seed', default=int(1000 * time.time()), help='Random seed (uses system time by default)')
    parser.add_option('-D', '--default_CONFIG', dest='default', default=False, action="store_true", help='Use default CONFIGuration file.')
    
    config = parse_sysargs(parser)
    return config

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
                         sequence_length)
    title = title.lower()

    print(title)
    
    combine_args = (CONFIG.background_path,
                    CONFIG.output_path,
                    CONFIG.font_path)
    post_args = (CONFIG.output_path,
                 CONFIG.post_to_facebook)
    # Try/Except is a basic way of keeping the bot up
    # while the image or text can fail to generate
    # try:
    try:
        make_background()
    except:
        title = "failed"
        pass
    combine_image(title, *combine_args)
    post_to_facebook(title, *post_args)

    log_image()

    # except:
    #     print("Background failed to generate")
    #     combine_image("failed", *combine_args)
    #     post_to_facebook("failed", *post_args)
    log_title(CONFIG.seed, title)

if __name__ == '__main__':

    if get_config("restart"):
        CONFIG = load_default_config()
        restart()

    delay = CONFIG.post_delay_seconds
    schedule.every(delay).seconds.do(job).run()
    
    while True:
        schedule.run_pending()
        time.sleep(1)
