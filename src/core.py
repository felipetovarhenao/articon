from PIL import Image, ImageDraw
from gamut.data import KDTree
from random import randint, choice, random
from .config import RESAMPLING_METHOD
from .utils import resize_img, get_dominant_color
import numpy as np
from typing import Callable
from collections.abc import Iterable
from typing_extensions import Self
import os
from gamut.config import CONSOLE


class PoissonDiskSampler:
    """
    A Poisson disk sampler creates a random distribution of points in a 2-dimensional space, while ensuring
    that each point will have a user-defined minimum distance to every neighboring point.
    """

    def __init__(
            self,
            width: int,
            height: int,
            radius: int | float = 10,
            k: int = 10,
            distance_func: Callable | None = None,
            sample_func: Callable | None = None,
    ) -> None:
        self.radius = radius
        self.k = k
        self.width = width
        self.height = height
        self.distance_func = distance_func or self.euclidean_distance
        self.sample_func = sample_func or (lambda _: None)

        self.cell_size = self.radius**(0.5)

        self.cols = int(width // self.cell_size)
        self.rows = int(height // self.cell_size)

        self.grid = np.empty(shape=(self.cols*self.rows, 2))
        self.grid.fill(-1)
        self.active = []
        self.__populate()

    @staticmethod
    def euclidean_distance(a, b) -> np.ndarray:
        """
        Static method to compute euclidean distance, as a default distance function
        """
        return ((a - b) ** 2).sum() ** 0.5

    def __initialize(self) -> None:
        """ Inserts the center coordinates of the 2D space in the ``grid`` and ``active`` arrays """
        point = np.array([self.width, self.height]) / 2
        i, j = (point // self.cell_size).astype('int64')
        self.grid[int(i + j * self.cols)] = point
        self.active.append(point)

    def __generate_random_point(self) -> np.ndarray:
        """ Generates a evenly-distributed random point between ``radius`` and ``radius*2`` from the origin """
        theta, radius = (np.random.rand(2)**np.array([1, 0.5]))*np.array([np.pi*2, self.radius])
        pt = np.array([np.cos(theta), np.sin(theta)])
        return pt * radius + pt * self.radius

    def __populate(self) -> None:
        # start with center coordinates and insert in self.grid and self.active
        self.__initialize()
        CONSOLE.counter.reset('Matching target segments:')
        # begin search
        while self.active:
            pos_idx = randint(0, len(self.active) - 1)
            pos = self.active[pos_idx]
            found = False
            for _ in range(self.k):
                sample = self.__generate_random_point() + pos
                col, row = (sample // self.cell_size).astype('int64')
                grid_index = int(col + row * self.cols)

                # if coordinates are valid and point is empty, check distance to neighbors
                if 0 <= col < self.cols and 0 <= row < self.rows and -1 in self.grid[grid_index]:
                    far_enough = True
                    for i in range(-1, 2):
                        if not far_enough:
                            break
                        for j in range(-1, 2):
                            idx = col + i + (row + j) * self.cols
                            if not 0 <= idx < len(self.grid):
                                continue
                            neighbor = self.grid[idx]
                            if -1 not in neighbor:
                                d = self.distance_func(sample, neighbor)
                                if d < self.radius:
                                    far_enough = False
                                    break

                    # if all existing neighbors are far enough, register point
                    if far_enough:
                        found = True
                        self.grid[grid_index] = sample
                        self.active.append(sample)
                        self.sample_func(sample)
                        CONSOLE.counter.next()
                        break

            # remove point if useless
            if not found:
                self.active.pop(pos_idx)

    def show(self, point_size=1):
        img = Image.new(mode='RGBA', size=(self.width, self.height), color=(0, 0, 0))
        draw = ImageDraw.Draw(img)
        for x, y in self.grid:
            if -1 in (x, y):
                continue
            draw.ellipse((x-point_size, y-point_size, x+point_size, y+point_size), fill=(255, 255, 255, 255))
        img.show()


class ImageCorpus:

    def __init__(self,
                 images: Iterable | None = None,
                 leaf_size: int = 10,
                 feature_extraction_func: Callable | None = None,
                 error_tolerance: float = 0.25,
                 alpha_threshold: int = 127) -> None:
        self.images = images or []
        self.tree = KDTree(leaf_size=leaf_size)
        self.feature_extraction_func = feature_extraction_func or self.__get_feature_extraction_func(
            error_tolerance, alpha_threshold)
        self.tree.build(data=[self.feature_extraction_func(img) for img in self.images], vector_path='features')

    @classmethod
    def read(cls, folder_path, selection_filter: Callable | None = None, size: Iterable | None = None, *args, **kwargs) -> Self:
        images = []
        for root, _, files in os.walk(folder_path):
            for file in files:
                ext = os.path.splitext(file)[1]
                file_path = os.path.join(root, file)
                if ext.lower() not in ['.jpeg', '.jpg', '.png']:
                    continue
                img = Image.open(file_path)
                if not selection_filter or (selection_filter and selection_filter(img)):
                    if size:
                        img = resize_img(img, size)
                    images.append(img)
        corpus = cls(images=images, *args, **kwargs)
        return corpus

    def plot(self, error_tolerance: float = 0.3):
        count = len(self.images)
        cols = rows = int(count**0.5)
        rows += 1
        cell_size = 30
        gap = cell_size // 2
        w, h = cols * 2 * (cell_size+gap), rows * (cell_size + gap)
        size = (cell_size, cell_size)
        canvas = Image.new(mode='RGBA', size=(w, h), color=(0, 0, 0, 0))
        for i in range(rows):
            for j in range(cols):
                idx = i * cols + j
                left, top = (i * cell_size * 2) + (gap * i * 2), j * cell_size + (gap * j)
                if idx >= count:
                    break
                img = self.images[idx].resize(size=(cell_size, cell_size))
                color = tuple(get_dominant_color(img, error_tolerance))
                cell = Image.new(mode='RGBA', size=size, color=(*color, 255))
                canvas.paste(img, box=(left, top))
                canvas.paste(cell, box=(left+cell_size, top))
        canvas.show()

    def __get_feature_extraction_func(self, error_tolerance, alpha_treshold):
        def feature_extraction_func(img):
            return {
                'image': img,
                'features': get_dominant_color(img, error_tolerance, alpha_treshold)[:3]
            }
        return feature_extraction_func


class ImageMosaic:
    """
    An image mosaic represents a reconstruction of a target image using an image corpus.
    """

    def __init__(
            self, target: str, corpus: ImageCorpus, radius: int = 10, k: int = 10, scale_target: float = 1.0, num_choices: int = 1,
            target_mix: float = 0.0) -> None:
        self.target = Image.open(target).convert('RGBA')
        if scale_target != 1.0:
            self.target = self.target.resize(
                size=(int(self.target.width * scale_target),
                      int(self.target.height * scale_target)),
                resample=RESAMPLING_METHOD)
        self.corpus = corpus
        self.mosaic = Image.new(mode='RGBA', size=(self.target.width, self.target.height), color=(0, 0, 0, 0))
        self.radius = (np.cos(radius), np.sin(radius))
        self.sampler = PoissonDiskSampler(
            width=self.target.width, height=self.target.height, radius=radius, k=k,
            sample_func=self.__get_feature_extraction_func(radius, num_choices, target_mix))
        if target_mix > 0.0:
            self.target.paste(self.mosaic, mask=self.mosaic.convert('LA'))
            self.mosaic = self.target

    def __get_feature_extraction_func(self, radius, num_choices, target_mix):
        def sample_func(point) -> None:
            x, y = point
            left, top, right, bottom = np.array([x-radius, y-radius, x+radius, y+radius]).astype('int64')
            segment = self.target.crop(box=((left, top, right, bottom)))
            if random() < target_mix:
                return
            data = self.corpus.feature_extraction_func(segment)
            matches = self.corpus.tree.knn(x=data['features'], vector_path='features', first_n=num_choices)
            best_match = choice(matches)['value']['image']
            best_match = best_match.rotate(randint(0, 360), resample=Image.Resampling.BICUBIC, expand=1)
            box = tuple((point - np.array(best_match.size) // 2).astype('int64'))
            self.mosaic.paste(best_match, box=box, mask=best_match)
        return sample_func

    def save(self, path: str = 'mosaic', *args, **kwargs) -> None:
        self.mosaic.save(path, *args, **kwargs)

    def show(self) -> None:
        self.mosaic.show()

    def resize(self, *args, **kwargs):
        self.mosaic.resize(*args, **kwargs)
