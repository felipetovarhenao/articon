from __future__ import annotations
from .image import Image, ImageDraw
from .utils import euclidean_distance
from random import randint
from .config import COUNTER
import numpy as np
from typing import Callable


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
            sample_func: Callable | None = None) -> None:
        self.radius = radius
        self.k = k
        self.width = width
        self.height = height
        self.distance_func = distance_func or euclidean_distance
        self.sample_func = sample_func or (lambda _: None)

        self.cell_size = self.radius / np.sqrt(2)

        self.cols = int(width // self.cell_size)
        self.rows = int(height // self.cell_size)

        self.grid = np.empty(shape=(self.cols*self.rows, 2))
        self.grid.fill(-1)
        self.active = []
        self.ordered = []
        self.__populate()

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
        if self.sample_func:
            COUNTER.reset('Finding best matches:')

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
                            if not far_enough:
                                break
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
                        self.ordered.append(sample)
                        self.sample_func(sample)
                        COUNTER.next()
                        break

            # remove point if useless
            if not found:
                self.active.pop(pos_idx)
        COUNTER.finish()

    def get_points(self):
        """ Returns points """
        return self.ordered

    def show(self, point_size: int = 1) -> None:
        """ Displays sampled points """
        img = Image.new(mode='RGBA', size=(self.width, self.height), color=(0, 0, 0))
        draw = ImageDraw.Draw(img)
        for x, y in self.ordered:
            draw.ellipse((x-point_size, y-point_size, x+point_size, y+point_size), fill=(255, 255, 255, 255))
        img.show()
