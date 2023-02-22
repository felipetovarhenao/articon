from __future__ import annotations
from .image import Image, ImageStat, ImageChops, ImageOps, ImageDraw
import numpy as np
from .config import RESAMPLING_METHOD
from collections.abc import Iterable
import cv2


def get_dominant_color(img: Image.Image, error_tolerance: float = 0.25, alpha_threshold: int = 127, n_colors: int = 16) -> Iterable:
    """ Finds the most prevalent color in an image, """

    # quantize image into n_colors
    im = img.quantize(n_colors, method=Image.Quantize.FASTOCTREE)

    # get num of active pixels
    arr = np.array(img)[:, :, 3].reshape(img.size[0]*img.size[1])
    num_pixels = np.where(arr > alpha_threshold, 1, 0).sum()

    try:
        mean = ImageStat.Stat(im.convert('RGB'), im.convert('L')).mean
    except:
        mean = (0, 0, 0)
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
    density = 0

    # look for color that passes error and alpha thresholds
    for i in range(len(dominant_sorted)):
        pixel_count, (pixel, alpha) = dominant_sorted[i]
        if alpha < alpha_threshold:
            continue
        col = palette[pixel]
        error = np.sum((col[:3]-mean[:3])**2)**0.5
        if error < min_error:
            best_candidate = col
            min_error = error
            density = pixel_count/num_pixels
        if error > error_threshold:
            continue
        found = True
        break

    return col if found else best_candidate, density


def rgb2hex(r: int, g: int, b: int) -> str:
    """ RGB to HEX conversion """
    return '#{:02x}{:02x}{:02x}'.format(r, g, b)


def resize_img(img: Image.Image, size: Iterable | float | int) -> Image.Image:
    """
    Resizes image while preserving aspect ratio.
    If size is an iterable, it fits centered image within specified size, down- or up-sizing if needed, and pads as necessary.
    If size is a float/int < 10, the value is interpreted as a rescaling factor
    If size is a float/int >= 10, the value is interpreted as the length of the longest edge.
    """

    in_size = img.size

    # resolve intended resizing
    if isinstance(size, Iterable):
        # 1) explicit
        out_size = size
    elif isinstance(size, float | int):
        if size < 10:
            # 2) scaling factor
            factor = size
        else:
            # 3) max side length
            factor = size/max(*in_size)
        out_size = tuple(int(x*factor) for x in in_size)

        # normal resizing method
        return img.resize(out_size, RESAMPLING_METHOD)
    else:
        raise TypeError(f'{size} must be either an int, float, or iterable.')

    # do nothing if size is the same
    if in_size[0] == out_size[0] and in_size[1] == out_size[1]:
        return img

    img_copy = img.copy()

    if in_size[0] > out_size[0] or in_size[1] > out_size[1]:
        # downscaling
        img_copy.thumbnail(out_size, RESAMPLING_METHOD)
    else:
        # upscaling
        img_copy = resize_img(img_copy, max(*out_size))

    # update image size
    in_size = img_copy.size

    # apply padding and center
    thumb = img_copy.crop((0, 0, out_size[0], out_size[1]))

    offset_x = max((out_size[0] - in_size[0]), 0) // 2
    offset_y = max((out_size[1] - in_size[1]), 0) // 2
    return ImageChops.offset(thumb, xoffset=offset_x, yoffset=offset_y)


def pixelate_image(img: Image.Image, pixel_size: int, error_tolerance: float = 0.25, alpha_threshold: int = 127) -> Image.Image:
    cols, rows = (np.array(img.size) // pixel_size).astype('int64')
    canvas = Image.new(mode='RGBA', size=img.size, color=(0, 0, 0, 0))
    for i in range(rows):
        for j in range(cols):
            left, top = pixel_size*i, pixel_size*j
            box = (left, top, left + pixel_size, top + pixel_size)
            seg = img.crop(box)
            rgba = get_dominant_color(seg, error_tolerance, alpha_threshold)[0]
            col = Image.new(mode='RGBA', size=(pixel_size, pixel_size), color=tuple((*rgba, 255)))
            canvas.paste(col, box=(left, top))
    return canvas


def xy_random(x: float | int, y: float | int) -> float:
    """ Deterministically generates random value in the range of [0, 1) from two values"""
    return np.sin(np.dot([x, y], [12.98931, 78.2357]))*43758.5453123 % 1.0


def add_color_layer(im: Image.Image, rgb: Iterable) -> Image.Image:
    img = im.convert('LA').convert('RGBA')
    layer = Image.new('RGBA', im.size, tuple((*rgb, 255)))
    return ImageChops.overlay(img, layer)


def create_image_palette(bits: int = 8, func: None = None) -> Image.Image:
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


def bgr2rgb(arr: np.ndarray) -> np.ndarray:
    """ Converts a BGR formatted numpy array into RGB """
    return cv2.cvtColor(arr, cv2.COLOR_BGR2RGB)


def rgb2bgr(arr: np.ndarray) -> np.ndarray:
    """ Converts an RGB formatted numpy array into BGR """
    return cv2.cvtColor(arr, cv2.COLOR_RGB2BGR)


def write_frame(writer: cv2.VideoWriter, frame: Image.Image) -> None:
    writer.write(rgb2bgr(np.array(frame)))
