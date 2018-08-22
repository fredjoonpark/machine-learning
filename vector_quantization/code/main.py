# basics
import os
import pickle
import argparse
import matplotlib.pyplot as plt
import numpy as np

from quantize_image import ImageQuantizer
from skimage.io import imread, imshow, imsave

def load_dataset(filename):
    with open(os.path.join('..','data',filename), 'rb') as f:
        return pickle.load(f)

# A simple image-compression algorithm (vector quantization)
# using k-means clustering
if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('-m','--module', required=True)

    io_args = parser.parse_args()
    module = io_args.module

    if module == '1':
        # sample image of a mandrill (lol)
        img = imread(os.path.join("..", "data", "mandrill.jpg"))

        # b-bit colour representation
        for b in [1,2,4,6]:
            quantizer = ImageQuantizer(b)
            q_img = quantizer.quantize(img)
            d_img = quantizer.dequantize(q_img)

            plt.figure()
            plt.imshow(d_img)
            fname = os.path.join("..", "figs", "{}_bit_image.png".format(b))
            plt.savefig(fname)
            print("Figure saved as '%s'" % fname)
