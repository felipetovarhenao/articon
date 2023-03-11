from .image import Image
from .features import get_dominant_color
from typing import Callable


def make_density_filter(min_density: float = 0.35, **kwargs) -> Callable:
    """
    Returns a selection filter callback, which returns a boolean value based on whether
    an input image's average color has enough presence in the image (pixel density).

    NOTE: kwargs are passed to the get_dominant_color function
    """

    def filter(img: Image.Image) -> bool:
        _, density = get_dominant_color(img, **kwargs)
        if density < min_density:
            return False
        return True

    return filter
