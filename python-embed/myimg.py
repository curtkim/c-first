import numpy
from PIL import Image

def shape(img):
    print(type(img))

    pil_im = Image.fromarray(img)
    print(pil_im.mode)
    pil_im.save("myimg.png")
    return img.shape