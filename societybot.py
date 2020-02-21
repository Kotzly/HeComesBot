import numpy as np, optparse, random, time, schedule
from PIL import Image, ImageDraw, ImageFont
import markovify
import facebook

def makeText():
    # Get raw text as string.
    with open(r"C:\Users\Administrator\Desktop\societyBot-master\quotes.txt", encoding='utf8') as f:
       text = f.read()

    # Build the model.
    text_model1 = markovify.NewlineText(text, state_size=2)
    

    return text_model1.make_short_sentence(50)


def MakeBackground():
     # Parse command-line options
    parser = optparse.OptionParser()
    parser.add_option('-o', '--output', dest='outputPath', help='Write output to FILE', metavar='FILE')
    parser.add_option('-d', '--dims', dest='dims', default='512x512', help='Image width x height, e.g. 320x240')
    parser.add_option('-s', '--seed', dest='seed', default=int(1000 * time.time()), help='Random seed (uses system time by default)')
    options, _ = parser.parse_args()

    dX, dY = (int(n) for n in options.dims.lower().split('x'))
    try:
        options.seed = int(options.seed)
    except ValueError:
        pass
    random.seed(options.seed)
    if not options.outputPath:
        options.outputPath = r'C:\Users\Administrator\Desktop\societyBot-master\output.png'

    # Generate x and y images, with 3D shape so operations will correctly broadcast.
    xArray = np.linspace(0.0, 1.0, dX).reshape((1, dX, 1))
    yArray = np.linspace(0.0, 1.0, dY).reshape((dY, 1, 1))

    # Adaptor functions for the recursive generator
    # Note: using python's random module because numpy's doesn't handle seeds longer than 32 bits.
    def randColor(): return np.array([random.random(), random.random(), random.random()]).reshape((1, 1, 3))
    def xVar(): return xArray
    def yVar(): return yArray
    def safeDivide(a, b): return np.divide(a, np.maximum(b, 0.001))

    # Recursively build an image using a random function.  Functions are built as a parse tree top-down,
    # with each node chosen randomly from the following list.  The first element in each tuple is the
    # number of required recursive calls and the second element is the function to evaluate the result.
    functions = (
            (0, randColor),
            (0, xVar),
            (0, yVar),
            (1, np.sin),
            (1, np.cos),
            (2, np.add),
            (2, np.subtract),
            (2, np.multiply),
            (2, safeDivide),
        )

    depthMin = 5
    depthMax = 15

    def buildImg(depth = 0):
        funcs = [f for f in functions if
                    (f[0] > 0 and depth < depthMax) or
                    (f[0] == 0 and depth >= depthMin)]
        nArgs, func = random.choice(funcs)
        args = [buildImg(depth + 1) for n in range(nArgs)]
        return func(*args)

    img = buildImg()

    # Ensure it has the right dimensions
    img = np.tile(img, (dX / img.shape[0], dY / img.shape[1], 3 / img.shape[2]))

    # Convert to 8-bit, send to PIL and save
    img8Bit = np.uint8(np.rint(img.clip(0.0, 1.0) * 255.0))
    
    Image.fromarray(img8Bit).save(options.outputPath)
    print('Seed %s; wrote output to %s' % (repr(options.seed), options.outputPath))


def TextWrap(text, font, max_width):

    lines = []

    #if the width is smaller than the image width, add to lines array
    if font.getsize(text)[0] <= max_width:
        lines.append(text)
    else:
        #split text by space to find words
        words = text.split(' ')
        i = 0

        #add each word to a line while the line is shorter than the image
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


def CombineImage(text):


    #make the new image
    img = Image.open(r"C:\Users\Administrator\Desktop\societyBot-master\output.png")
    width, height = img.size
    #setup d for drawing
    d = ImageDraw.Draw(img)
    #set image size
    image_size = (width, height)
    #set the colour of text & outline
    textcolour = (0,0,0)
    textOutline = (255, 255, 255)
    #create the font and set text size
    fnt = ImageFont.truetype(r"C:\Users\Administrator\Desktop\societyBot-master\zalgo.ttf", 40, encoding="unic")

    draw = ImageDraw.Draw(img)
    #Set outline amount, sets how many times we loop
    outlineAmount = 3
    

    #set the height of each line
    line_height = fnt.getsize('hg')[1]
    x = random.randint(0, 50)
    y = random.randint(0, 150)

    
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
                d.text((x,y), lines, fill=textcolour, font=fnt)
                
                

                y = y + line_height
    
    else:
        d.text((x,y), text, font=fnt, fill=textcolour)


    img.save(r"C:\Users\Administrator\Desktop\societyBot-master\outputWithText.png", optimize=True)
    print('Text written')


def postToFacebook(post_text):
    token = ""
    graph = facebook.GraphAPI(token)
    post_id = graph.put_photo(image = open(r"C:\Users\Administrator\Desktop\societyBot-master\outputWithText.png", 'rb'), message=post_text)['post_id']
    
    print(f"Success in uploading {post_id} to facebook")


def job():
    title = makeText()
    print(title)

    try:
        MakeBackground()
        CombineImage(title.lower())
        postToFacebook(title.lower())

    except:
        print("Backgrounf failed to generate")

        CombineImage("failed")

        postToFacebook("failed")


if __name__ == '__main__':

    schedule.every(30).minutes.do(job).run()
    
    while True:
        schedule.run_pending()
        time.sleep(1)
