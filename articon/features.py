from .image import Image, ImageStat
from .config import MAX_RGB_DISTANCE, RGB_WEIGHTS
from collections.abc import Iterable
import numpy as np


def get_dominant_color(img: Image.Image, error_tolerance: float = 0.25, alpha_threshold: int = 127, n_colors: int = 16) -> Iterable:
    """ Finds the most prevalent color in an image """

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
        error = ((RGB_WEIGHTS*((col[:3] - mean[:3]) ** 2)).sum() ** 0.5) / MAX_RGB_DISTANCE
        if error < min_error:
            best_candidate = col
            min_error = error
            density = pixel_count/num_pixels
        if error > error_threshold:
            continue
        found = True
        break

    return col if found else best_candidate, density
