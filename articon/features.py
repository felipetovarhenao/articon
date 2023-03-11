from .image import Image
from .config import RGB_WEIGHTS
from collections.abc import Iterable
import numpy as np


def get_dominant_color(img: Image.Image, max_distance: int = 30, alpha_threshold: int = 127) -> Iterable:
    """
    Finds the most dominant color in an image, and the pixel density (0-1) for that color

    img: Image
        A PIL image.
    max_distance: int | float = 30
        Maximum distance a color can be from the mean, in order to count toward pixel count.
    alpha_threshold: int = 127
        The minumum alpha value for a pixel to be considered as active.
    """

    # converts to rgba if needed
    image = img if img.mode == 'RGBA' else img.convert("RGBA")

    # convert image to numpy array and flatten
    flat_image = np.array(image).reshape(image.size[0]*image.size[1], 4)

    # split rgb from alpha
    rgb_array, alpha_array = flat_image[:, :3], flat_image[:, 3]

    # create mask from active pixels and use it as weights to compute average color
    pixel_mask = np.where(alpha_array > alpha_threshold, 1, 0)
    num_pixels = pixel_mask.sum()

    # return with 0 density if no pixe
    if not num_pixels:
        return np.array([0, 0, 0]), 0.0
    mean = np.average(a=rgb_array, axis=0, weights=pixel_mask)

    # convert to alpha palette
    im = img.convert('PA')

    # reshape palette into RGB subarrays
    palette = np.array(im.getpalette())
    palette = palette.reshape((len(palette)//3, 3))

    # initialize dominant color pixel count
    pixel_count = 0

    # iterate through every unique color in image and get pixel count if it passes conditions
    for count, (index, alpha) in im.getcolors():

        # reject if too transparent
        if alpha < alpha_threshold:
            continue
        rgb = palette[index][:3]
        dist = ((RGB_WEIGHTS*((rgb - mean) ** 2)).sum() ** 0.5)

        # reject if too different from mean
        if dist > max_distance:
            continue

        # include in pixel count
        pixel_count += count

    # return mean color and pixel density
    return mean.astype('int64'), pixel_count / num_pixels
