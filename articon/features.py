from __future__ import annotations
from .image import Image
from random import randint, choice, random, Random
from .config import RESAMPLING_METHOD, COUNTER, BAR
from .utils import resize_img, get_dominant_color
import numpy as np
from typing import Callable
from collections.abc import Iterable
from typing_extensions import Self
import os
from sklearn.neighbors import KDTree
import cv2
import subprocess
from .samplers import PoissonDiskSampler


class IconCorpus:
    """
    An icon corpus represents a collection of small images (e.g., 60x60 pixels), ideally diverse in colors, that can be 
    used to construct mosaics based on a given target.
    """

    def __init__(self,
                 images: Iterable,
                 leaf_size: int = 10,
                 feature_extraction_func: Callable | None = None,
                 error_tolerance: float = 0.25,
                 alpha_threshold: int = 127,
                 precomputed_features: Iterable | None = None) -> None:
        if precomputed_features and not feature_extraction_func:
            raise ValueError("When providing precomputed features, you must also provide a feature extraction function.")
        self.images = images
        self.feature_extraction_func = feature_extraction_func or IconCorpus.__get_feature_extraction_func(
            error_tolerance, alpha_threshold)
        features = precomputed_features or [self.feature_extraction_func(img) for img in self.images]
        self.tree = KDTree(features, leaf_size=leaf_size)

    @classmethod
    def read(cls,
             source: str,
             selection_filter: Callable | None = None,
             size: Iterable | None = None,
             error_tolerance: float = 0.25,
             alpha_threshold: int = 127,
             feature_extraction_func: Callable | None = None,
             random_loading_seed: int | None = None,
             *args,
             **kwargs) -> Self:
        """ Builds a corpus from a folder of images, recursively loading any .jpeg, .jpg, or .png file within. """
        images = []
        features = []
        feature_func = feature_extraction_func or IconCorpus.__get_feature_extraction_func(error_tolerance, alpha_threshold)
        COUNTER.reset('Building corpus from images:')
        for root, _, files in os.walk(source):
            cls.__handle_random_loading(files, random_loading_seed)
            for file in files:
                ext = os.path.splitext(file)[1]
                file_path = os.path.join(root, file)
                if ext.lower() not in ['.jpeg', '.jpg', '.png']:
                    continue
                img = Image.open(file_path)
                if not selection_filter or (selection_filter and selection_filter(img, file)):
                    if size:
                        img = resize_img(img, size)
                    im = img.convert("RGBA")
                    images.append(im)
                    features.append(feature_func(im))
                    COUNTER.next()
        COUNTER.finish()
        if not images:
            raise ValueError(
                'No images to process. If using a selection filter function, make sure that at least one image passes the test')
        corpus = cls(images=images, precomputed_features=features, feature_extraction_func=feature_func, *args, **kwargs)
        return corpus

    @staticmethod
    def __handle_random_loading(files: Iterable, seed: int | None) -> None:
        """ Checks if random loading seed is valid and warns user """
        if not isinstance(seed, int | None):
            raise TypeError(f"{seed} must be an integer or NoneType. Ignoring value...")
        elif seed is not None:
            Random(seed).shuffle(files)

    def show(self, error_tolerance: float = 0.25) -> None:
        """ Display every image in corpus along with its associated dominant color """
        count = len(self.images)
        cols = rows = int(count**0.5)
        cols += 1
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
                color = tuple(get_dominant_color(img, error_tolerance))[0]
                cell = Image.new(mode='RGBA', size=size, color=(*color, 255))
                canvas.paste(img, box=(left, top))
                canvas.paste(cell, box=(left+cell_size, top))
        canvas.show(title='corpus')

    @classmethod
    def __get_feature_extraction_func(cls, error_tolerance: float | int, alpha_treshold: int) -> Callable:
        def feature_extraction_func(img: Image.Image) -> Iterable:
            color, density = get_dominant_color(img, error_tolerance, alpha_treshold)
            return (*color, density)
        return feature_extraction_func


class IconMosaic:
    """
    An image mosaic represents a reconstruction of a target image using an image corpus.
    """

    def __init__(
            self,
            target: str | Image.Image,
            corpus: IconCorpus,
            radius: int = 10,
            k: int = 10,
            scale_target: float = 1.0,
            num_choices: int = 1,
            target_mix: float = 0.0,
            keep_frames: bool = False,
            frame_hop_size: int | None = None) -> None:

        self.target = Image.open(target).convert('RGBA') if isinstance(target, str) else target

        # resize target if needed
        if scale_target != 1.0:
            self.target = self.target.resize(
                size=(int(self.target.width * scale_target),
                      int(self.target.height * scale_target)),
                resample=RESAMPLING_METHOD)

        self.corpus = corpus
        self.frames = []
        self.mosaic = Image.new(mode='RGBA', size=self.target.size, color=(0, 0, 0, 0))

        # get grid cell size
        cell_size = int((radius / np.sqrt(2)))

        # initialize variables for periodic frame storage
        self.frame_counter = 0
        self.frame_counter_modulo = frame_hop_size or int(max(1, cell_size))

        # build mosaic using poisson disk sampler
        self.sampler = PoissonDiskSampler(
            width=self.target.width, height=self.target.height, radius=radius, k=k,
            sample_func=self.__get_sample_func(cell_size, num_choices, target_mix, keep_frames))

        # paste mosaic on top of target mix if used
        if target_mix > 0.0:
            tmp = self.target.copy()
            tmp.paste(self.mosaic, mask=self.mosaic.convert('LA'))
            self.mosaic = tmp

    def __get_sample_func(self, xy_offset: int, num_choices: int, target_mix: float, keep_frames: bool):
        def sample_func(point: Iterable) -> None:
            x, y = point
            left, top, right, bottom = np.array([x-xy_offset, y-xy_offset, x+xy_offset, y+xy_offset]).astype('int64')
            segment = self.target.crop(box=((left, top, right, bottom)))
            if random() < target_mix:
                return
            feature = self.corpus.feature_extraction_func(segment)
            indexes = self.corpus.tree.query([feature], k=num_choices)[1][0]
            matches = [self.corpus.images[i] for i in indexes]
            best_match = choice(matches)
            best_match = best_match.rotate(randint(0, 360), resample=Image.Resampling.BICUBIC, expand=1)
            box = tuple((point - np.array(best_match.size) // 2).astype('int64'))
            self.mosaic.paste(best_match, box=box, mask=best_match)
            if keep_frames and self.frame_counter % self.frame_counter_modulo == 0:
                self.frames.append(self.mosaic.copy())
            self.frame_counter += 1
        return sample_func

    def save_as_video(self,
                      path: str = 'mosaic.mp4',
                      frame_rate: int = 60,
                      background_image: Image.Image | None = None,
                      max_duration: int = 5,
                      open_file: bool = False) -> None:
        if not self.frames:
            raise ValueError(
                f"To use the save_as_video method you must set the keep_frames argument to True when creating an instance of the {self.__class__.__name__} class")

        full_duration = len(self.frames) / frame_rate
        hop_size = int(max(1, round(full_duration / max_duration)))
        final_duration = full_duration / hop_size
        final_frame_count = final_duration * frame_rate

        def write_frame(video, frame):
            video.write(cv2.cvtColor(np.array(frame), cv2.COLOR_RGB2BGR))
        fourcc = cv2.VideoWriter_fourcc(*'avc1')
        video = cv2.VideoWriter(path, fourcc, frame_rate, self.mosaic.size)
        bg = resize_img(background_image, self.mosaic.size).convert('RGBA') if background_image else None

        BAR.reset('Writing video frames', max=final_frame_count+1, item='frames')
        if bg:
            write_frame(video, bg)
            BAR.next()
        for frame in self.frames[::hop_size]:
            if bg:
                im = bg.copy()
                im.paste(frame, box=(0, 0), mask=frame)
            else:
                im = frame
            write_frame(video, im)
            BAR.next()
        video.release()
        if open_file:
            subprocess.run(['open', path])
        BAR.finish()

    def save(self, path: str = 'mosaic', *args, **kwargs) -> None:
        self.mosaic.save(path, *args, **kwargs)

    def show(self) -> None:
        self.mosaic.show(title='mosaic')

    def resize(self, *args, **kwargs) -> None:
        self.mosaic.resize(*args, **kwargs)
