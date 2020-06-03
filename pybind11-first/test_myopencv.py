import numpy as np
import myopencv
from PIL import Image

im = myopencv.get_image()
print(type(im))
a = np.array(im, copy=False)

print(a.shape)
image = Image.fromarray(a)
image.save("myopencv.png")
