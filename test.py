import numpy as np
from PIL import Image, ImageOps

gt_fileName = './data/cleanchangemap.jpg'
im = Image.open((gt_fileName))
changedmap = ImageOps.expand(im, border=2)
img_labels = np.array(changedmap)


def binarize(image):
    labels_width = image.shape[0]
    labels_height = image.shape[1]

    for i in range(labels_width):
        for j in range(labels_height):
            if image[i,j] >=150:
                image[i,j] = 2;
            else:
                image[i,j] =1;
    return image

img_labels = binarize(img_labels)

filename = './data/diff.jpg'
img = np.array(Image.open(filename))

print(img_labels.shape)
print(img.shape)