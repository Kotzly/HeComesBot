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

def makeText(quotes_path, state_size=2, sequence_length=50):
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

def MakeBackground(config):
     # Parse command-line options
    
    dX, dY = config.dims

    random.seed(config.seed)

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
            if (n_args > 0 and depth < config.max_depth) or (n_args == 0 and depth >= config.min_depth):
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
        n_args, func = get_random_function(depth, p=config.personality)
        with open(config.save_path + f"{config.seed}_tree.txt", "a") as file:
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
    
    Image.fromarray(img_8bit).save(config.background_path)
    print('Seed {}; wrote output to {}'.format(config.seed, config.background_path))


# This is being worked on, as wrapped text won't draw an outline for some reason
# as such, the max text size is 50 chars so it does not have to wrap
def TextWrap(text, font, max_width):

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


def CombineImage(text, background_path="./background.png", output_path="./output.png", font_path="./zalgo.ttf"):

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

        WrappedText1 = TextWrap(text, fnt, image_size[0])

        for lines in WrappedText1:

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


def postToFacebook(post_text, filepath="./output.png", post=True):
    #obvs token is hidden
    if not post:
        return
    token = ""
    graph = facebook.GraphAPI(token)
    with open(filepath, 'rb') as image_file:
        post_id = graph.put_photo(image = image_file, message=post_text)['post_id']
    print(f"Success in uploading {post_id} to facebook")


def parse_cmd_args():
    parser = optparse.OptionParser()
    parser.add_option('-o', '--output', dest='background_path', default="./background.png", help='Write output to FILE', metavar='FILE')
    parser.add_option('-d', '--dims', dest='dims', default='512x512', help='Image width x height, e.g. 320x240')
    parser.add_option('-s', '--seed', dest='seed', default=int(1000 * time.time()), help='Random seed (uses system time by default)')
    parser.add_option('-D', '--default_config', dest='default', default=False, action="store_true", help='Use default configuration file.')
    
    config = parse_sysargs(parser)
    return config
    
def log_image(config):

    if not os.path.isdir(config.save_path):
        os.makedirs(config.save_path)

    with Image.open(config.output_path) as output_file:
        filename = config.output_path.split("/")[-1].split(".")[0]
        filename += "_" + str(config.seed)
        filename += ".png"
        history_path = join(config.save_path, filename)
        output_file.save(history_path, optimize=True)

    with Image.open(config.background_path) as background_file:
        filename = config.background_path.split("/")[-1].split(".")[0]
        filename += "_" + str(config.seed)
        filename += ".png"
        history_path = join(config.save_path, filename)
        background_file.save(history_path, optimize=True)

def log_title(seed, title):
    with open("./log.txt", "a") as log_file:
        timestamp = time.time()
        text = "{}\t{}\t{}\n".format(timestamp, seed, title)
        log_file.write(text)

def restart(config):
    if os.path.isdir(config.save_path):
        shutil.rmtree(config.save_path)
    if os.path.isfile("./log.txt"):
        os.remove("./log.txt")

def job():
    config = parse_cmd_args()
    title = None
    while title is None:
        sequence_length = np.random.randint(config.min_sequence_length, config.max_sequence_length + 1)
        title = makeText(config.quotes_path,
                         config.markov_model_state_size,
                         sequence_length)
    title = title.lower()

    print(title)
    
    combine_args = (config.background_path,
                    config.output_path,
                    config.font_path)
    post_args = (config.output_path,
                 config.post_to_facebook)
    # Try/Except is a basic way of keeping the bot up
    # while the image or text can fail to generate
    # try:
    try:
        MakeBackground(config)
    except:
        title = "failed"
        pass
    CombineImage(title, *combine_args)
    postToFacebook(title, *post_args)

    if config.save_path is not None:
        log_image(config)

    # except:
    #     print("Background failed to generate")
    #     CombineImage("failed", *combine_args)
    #     postToFacebook("failed", *post_args)
    log_title(config.seed, title)

if __name__ == '__main__':

    if get_config("restart"):
        config = load_default_config()
        restart(config)

    delay = get_config("post_delay_seconds")
    schedule.every(delay).seconds.do(job).run()
    
    while True:
        schedule.run_pending()
        time.sleep(1)
