import math

import numpy as np
from PIL import Image


def to_image(element):
    image_array = []
    batch = math.floor(math.sqrt(element.size()[0]))
    element = np.array(element.tolist())
    for i in range(1, batch):
        sub_array = []
        for j in range(1, batch):
            sub_array.append(np.uint8(element[i * batch + j] * 255))
        image_array.append(sub_array)
    image = np.block(image_array).T
    if image.shape[-1] == 1:
        image = image.T[0]
    im = Image.fromarray(image)
    return im
