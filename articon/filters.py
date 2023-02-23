from .image import Image
from .features import get_dominant_color
from .utils import euclidean_distance
from .config import RGB_WEIGHTS, MAX_RGB_DISTANCE
from typing import Callable
import numpy as np
from skimage.color import rgb2lab


def make_poisson_filter(min_density: float = 0.25, min_distance: float = 0.01, **kwargs) -> Callable:
    """
    Returns a selection filter function, which returns a boolean value based on whether
    an input image has a sufficiently dominant color and is sufficiently distinct from previously
    chosen colors.

    The returned function is meant to be passed to the selection_filter argument in the IconCorpus class

    NOTE: kwargs are passed to the get_dominant_color function
    """

    radius = min(1, abs(min_distance)) * MAX_RGB_DISTANCE
    cell_size = max(1, radius / np.sqrt(2))
    num_cells = int(256 // cell_size)
    grid = np.empty(shape=(num_cells, num_cells, num_cells, 3))
    grid.fill(-1)

    def filter(img: Image.Image) -> bool:
        col, density = get_dominant_color(img.convert("RGBA"), **kwargs)
        if density < min_density:
            return False

        col = col[:3]

        x, y, z = (col // cell_size).astype('int64')

        if 0 <= x < num_cells and 0 <= y < num_cells and 0 <= z < num_cells and -1 in grid[x, y, z]:
            far_enough = True
            for i in range(-1, 2):
                r = x+i
                if not far_enough:
                    break
                if not 0 <= r < num_cells:
                    continue
                for j in range(-1, 2):
                    g = y+j
                    if not far_enough:
                        break
                    if not 0 <= g < num_cells:
                        continue
                    for k in range(-1, 2):
                        b = z+k
                        if not 0 <= b < num_cells:
                            continue
                        neighbor = grid[r, g, b]
                        if -1 in neighbor:
                            continue
                        d = euclidean_distance(rgb2lab(col), rgb2lab(neighbor), w=RGB_WEIGHTS) / MAX_RGB_DISTANCE
                        if d < min_distance:
                            far_enough = False
                            break
            if far_enough:
                grid[x, y, z] = col
            return far_enough
        else:
            return False
    return filter


def make_naive_poisson_filter(min_density: float = 0.25, min_distance: float = 0.01, **kwargs):
    palette = []

    def filter(img: Image.Image) -> bool:
        col, density = get_dominant_color(img.convert("RGBA"), **kwargs)
        if density < min_density:
            return False

        col = col[:3]
        if not palette:
            palette.append(col)
            return True
        far_enough = True
        for c in palette:
            distance = euclidean_distance(col, c, w=(0.3, 0.59, 0.11)) / MAX_RGB_DISTANCE
            if distance < min_distance:
                far_enough = False
                break
        if far_enough:
            palette.append(col)
        return far_enough
    return filter
