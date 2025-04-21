import os
import numpy as np
import torch
import torch.nn as nn
from torchvision import datasets, transforms

from PIL import Image
import matplotlib.pyplot as plt

from scipy.fftpack import dct

class DCT(object):
    def __init__(self):
        self.BLOCK_HEIGHT = 8
        self.BLOCK_WIDTH = 8
        self.BLOCK_SIZE = (self.BLOCK_HEIGHT, self.BLOCK_WIDTH)

    def div_block(self, img, block_size):
        img_height = img.height
        img_width = img.width
        block_height = block_size[0]
        block_width = block_size[1]
        assert(img_height % block_height == 0)
        assert(img_width % block_width == 0)

        blocks = []
        for i in range(0,img_height,block_height):
            for j in range(0,img_width,block_width):
                box = (j, i, j+block_width, i+block_height)
                block = np.array(img.crop(box))
                blocks.append(block)
        return np.array(blocks)

    def dct2(self, array_2d):
        return dct(dct(array_2d.T, norm = 'ortho').T, norm = 'ortho')

    def _dct2(self, array_2d):
        array_2d = array_2d.reshape(64)
        a = dct(array_2d, norm='ortho').T
        return a

    def __call__(self, img):
        image = img
        blocks = self.div_block(image, self.BLOCK_SIZE)
        b_blocks, g_blocks, r_blocks = blocks[:, :, :, 0], blocks[:, :, :, 1], blocks[:, :, :, 2]
        test_blocks = (b_blocks + g_blocks + r_blocks) / 3
        result = np.array([self._dct2(test_block) for test_block in test_blocks])
        return torch.from_numpy(result.reshape(256, 64).T).float()

    def __repr__(self):
        return "Simply DCT. What do you expect?"

class DFT(object):
    def __init__(self):
        pass

    def __call__(self, freq):
        out = torch.fft.fft(freq, norm="ortho")
        return out

    def __repr__(self):
        return "Simply DFT. What do you expect?"

class Ycbcr_convert():
    def __init__(self):
        pass

    def __call__(self, img):
        return img.convert('YCbCr')

    def __repr__(self):
        return "Convert a PIL Image from RGB to YCbCr"

image_height_pixel, image_width_pixel = 224, 224
image_height_freq, image_width_freq = 128, 128
tform_pixel = transforms.Compose([
    transforms.Resize((image_height_pixel,image_width_pixel), interpolation=transforms.InterpolationMode.BICUBIC),
    transforms.ToTensor()
])
tform_freq = transforms.Compose([
    transforms.Resize((image_height_freq,image_width_freq)),
    Ycbcr_convert(),
    DCT(),
])