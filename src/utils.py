from PIL import Image, ImageStat, ImageChops, ImageOps, ImageDraw
import numpy as np
from .config import RESAMPLING_METHOD


def get_dominant_color(img, error_tolerance: float = 0.25, alpha_threshold: int = 127, n_colors: int = 16):
    """ Finds the most prevalent color in an image, """

    # quantize image into n_colors
    im = img.quantize(n_colors, method=Image.Quantize.FASTOCTREE)
    mean = ImageStat.Stat(im.convert('RGB'), im.convert('PA')).mean
    im = im.convert('PA')

    # sort colors based on pixel count
    dominant_sorted = sorted(im.getcolors(), key=lambda x: (-x[0], x[1][1]))

    # reshape palette into RGB subarrays and initialize best candidate
    palette = np.array(im.getpalette())
    palette = palette.reshape((len(palette)//3, 3))
    best_candidate = palette[dominant_sorted[0][1][0]]

    # set error threshold
    error_threshold = 255 * (1 - error_tolerance)
    min_error = 1000
    found = False
    col = None

    # look for color that passes error and alpha thresholds
    for i in range(len(dominant_sorted)):
        pixel, alpha = dominant_sorted[i][1]
        if alpha < alpha_threshold:
            continue
        col = palette[pixel]
        error = np.sum((col-mean)**2)**0.5
        if error < min_error:
            best_candidate = col
            min_error = error
        if error > error_threshold:
            continue
        found = True
        break

    return col if found else best_candidate


def rgb2hex(r, g, b):
    """ RGB to HEX conversion """
    return '#{:02x}{:02x}{:02x}'.format(r, g, b)


def resize_img(img, size):
    """ Resize image preserving aspect ratio """
    pad = False
    img_size = img.size
    x, y = img_size
    if x < size[0] or y < size[1]:
        pad = True
    img.thumbnail(size, RESAMPLING_METHOD)

    if pad:
        thumb = img.crop((0, 0, size[0], size[1]))

        offset_x = max((size[0] - img_size[0]), 0)
        offset_y = max((size[1] - img_size[1]), 0)
        return ImageChops.offset(thumb, xoffset=offset_x, yoffset=offset_y)

    else:
        return ImageOps.fit(img, size, method=RESAMPLING_METHOD, centering=(0.5, 0.5))


def pixelate_image(img, pixel_size, error_tolerance: float = 0.25, alpha_threshold: int = 127) -> None:
    cols, rows = (np.array(img.size) // pixel_size).astype('int64')
    canvas = Image.new(mode='RGBA', size=img.size, color=(0, 0, 0, 0))
    for i in range(rows):
        for j in range(cols):
            left, top = pixel_size*i, pixel_size*j
            box = (left, top, left + pixel_size, top + pixel_size)
            seg = img.crop(box)
            rgba = get_dominant_color(seg, error_tolerance, alpha_threshold)
            col = Image.new(mode='RGBA', size=(pixel_size, pixel_size), color=tuple((*rgba, 255)))
            canvas.paste(col, box=(left, top))
    return canvas


def add_color_layer(im, rgb):
    img = im.convert('LA').convert('RGBA')
    layer = Image.new('RGBA', im.size, tuple((*rgb, 255)))
    return ImageChops.overlay(img, layer)


def create_image_palette(bits: int = 8, func: None = None):
    images = []
    chan = np.linspace(0, 255, bits).astype('int64')
    for r in chan:
        for g in chan:
            for b in chan:
                if func:
                    im = func((r, g, b))
                else:
                    size = 60
                    im = Image.new(mode='RGBA', size=(size, size), color=(0, 0, 0, 0))
                    draw = ImageDraw.Draw(im)
                    draw.ellipse(xy=(0, 0, size, size), fill=(0, 0, 0, 0), outline=(r, g, b, 255), width=12)
                images.append(im)
    return images