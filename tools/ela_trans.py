import os.path
from PIL import Image, ImageChops, ImageEnhance

def ELA(filename):
    im = Image.open(filename).convert('RGB')
    resaved_filename = filename.split(".j")[0]+"_ELA.jpg"
    if not os.path.exists(resaved_filename):
        im.save(resaved_filename, 'JPEG', quality=90)
    resaved_im = Image.open(resaved_filename)

    ela_im = ImageChops.difference(im, resaved_im)
    extrema = ela_im.getextrema()
    max_diff = max([ex[1] for ex in extrema])
    if max_diff == 0:
        max_diff = 1
    scale = 255.0 / max_diff

    ela_im = ImageEnhance.Brightness(ela_im).enhance(scale)

    return ela_im


if __name__ == '__main__':
    filename = './ela_test_pic/A0btXodDR.jpg'
    resaved_filename = './ela_test_pic/A0btXodDR_ELA.jpg'
    im = Image.open(filename).convert('RGB')
    im.save(resaved_filename, 'JPEG', quality=90)
    resaved_im = Image.open(resaved_filename)

    ela_im = ImageChops.difference(im, resaved_im)
    extrema = ela_im.getextrema()
    max_diff = max([ex[1] for ex in extrema])
    if max_diff == 0:
        max_diff = 1
    scale = 255.0 / max_diff

    ela_im = ImageEnhance.Brightness(ela_im).enhance(scale)