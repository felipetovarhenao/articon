from __future__ import annotations
from .image import Image
from random import randint, choice, random, Random
from .config import COUNTER, BAR
from .filters import make_density_filter
from .utils import resize_img, get_dominant_color, xy_random, bgr2rgb, write_frame
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
                 alpha_threshold: int = 127,
                 precomputed_features: Iterable | None = None) -> None:
        if precomputed_features and not feature_extraction_func:
            raise ValueError("When providing precomputed features, you must also provide a feature extraction function.")
        self.images = images
        self.feature_extraction_func = feature_extraction_func or IconCorpus.get_feature_extraction_func(alpha_threshold)
        features = precomputed_features or [self.feature_extraction_func(img) for img in self.images]
        self.tree = KDTree(features, leaf_size=leaf_size)

    @classmethod
    def read(cls,
             source: str,
             selection_filter: Callable | None = make_density_filter(),
             size: Iterable | int | float | None = 60,
             alpha_threshold: int = 127,
             feature_extraction_func: Callable | None = None,
             random_loading_seed: int | None = None,
             *args,
             **kwargs) -> Self:
        """ Builds a corpus from a folder of images, recursively loading any .jpeg, .jpg, or .png file within. """
        images = []
        features = []
        feature_func = feature_extraction_func or IconCorpus.get_feature_extraction_func(alpha_threshold)
        COUNTER.reset('Building corpus from images:')
        for root, _, files in os.walk(source):
            cls.__handle_random_loading(files, random_loading_seed)
            for file in files:
                ext = os.path.splitext(file)[1]
                file_path = os.path.join(root, file)
                if ext.lower() not in ['.jpeg', '.jpg', '.png']:
                    continue
                img = Image.open(file_path)
                if not selection_filter or (selection_filter and selection_filter(img)):
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
        if seed is not None and not isinstance(seed, int):
            raise TypeError(f"{seed} must be an integer or NoneType. Ignoring value...")
        elif seed is not None:
            Random(seed).shuffle(files)

    def show(self) -> None:
        """ Display every image in corpus along with its associated dominant color """
        count = len(self.images)
        cols = rows = int(count**0.5) + 1
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
                img = self.images[idx]
                color = self.feature_extraction_func(img)[:3]
                cell = Image.new(mode='RGBA', size=size, color=(*color, 255))
                if len(color) != 3:
                    color = get_dominant_color(img)
                canvas.paste(resize_img(img, size=cell_size), box=(left, top))
                canvas.paste(cell, box=(left+cell_size, top))
        canvas.show(title='corpus')

    @classmethod
    def get_feature_extraction_func(cls, alpha_treshold: int) -> Callable:
        def feature_extraction_func(img: Image.Image) -> Iterable:
            color, density = get_dominant_color(img, alpha_treshold)
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
            size: float | int | None = None,
            num_choices: int = 1,
            target_mix: float = 0.0,
            keep_frames: bool = False,
            frame_hop_size: int | None = None) -> None:

        # load target as image vs from path string
        self.target = Image.open(target).convert('RGBA') if isinstance(target, str) else target

        # resize if needed
        if size:
            self.target = resize_img(self.target, size=size)

        self.corpus = corpus
        self.frames = []

        # initialize empty canvas
        self.mosaic = Image.new(mode='RGBA', size=self.target.size, color=(0, 0, 0, 0))

        # get grid cell size
        cell_size = int((radius / np.sqrt(2)))

        # initialize variables for periodic frame storage
        self.frame_counter = 0
        self.frame_counter_modulo = frame_hop_size or int(max(1, cell_size))

        # build mosaic using poisson disk sampler
        self.sampler = PoissonDiskSampler(width=self.target.width,
                                          height=self.target.height,
                                          radius=radius,
                                          k=k,
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

        def write_frame(video, frame):
            video.write(cv2.cvtColor(np.array(frame), cv2.COLOR_RGB2BGR))

        full_duration = len(self.frames) / frame_rate
        hop_size = int(max(1, round(full_duration / max_duration)))
        final_duration = full_duration / hop_size
        final_frame_count = final_duration * frame_rate

        video = cv2.VideoWriter(path, cv2.VideoWriter_fourcc(*'avc1'), frame_rate, self.mosaic.size)
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

    def get_image(self):
        return self.mosaic

    def get_sampler(self):
        return self.sampler

    def save(self, path: str = 'mosaic', *args, **kwargs) -> None:
        self.mosaic.save(path, *args, **kwargs)

    def show(self) -> None:
        self.mosaic.show(title='mosaic')

    def resize(self, *args, **kwargs) -> None:
        self.mosaic.resize(*args, **kwargs)


class AnimatedIconMosaic:
    """
    An animated mosaic represents a reconstruction of a target video using an image corpus.

    target: str
        Path of target video
    corpus: IconCorpus
        An instance of the `ImageCorpus` class.
    """

    def __init__(self, target: str, corpus: IconCorpus) -> None:
        self.reader = cv2.VideoCapture(target)
        self.corpus = corpus
        self.theta = 0

    def render(self,
               output: str,
               max_duration: float | int = None,
               size: Iterable = (120, 120),
               frame_rate: int = 24,
               radius: int = 10) -> None:
        """ Writes a video mosaic animation to disk """

        # get input target's frame info
        reader_frame_rate = self.reader.get(cv2.CAP_PROP_FPS)
        max_frames = self.reader.get(cv2.CAP_PROP_FRAME_COUNT)

        # get output duration in seconds
        total_duration = min(max_duration, max_frames / reader_frame_rate)

        # get max read/write number of frames
        max_read_frames = int(total_duration * reader_frame_rate)
        max_write_frames = min(int(total_duration * frame_rate), max_read_frames)

        # warn if target frame rate is lower than desired
        if frame_rate > reader_frame_rate:
            print(
                f'Warning: The target\'s original frame rate ({frame_rate} fps) is lower than the requested frame rate ({reader_frame_rate} fps), and will therefore be ignored. Rendering at {reader_frame_rate} fps...')

        # get first frame
        image = self.reader.read()[1]

        # load first frame
        im = Image.fromarray(bgr2rgb(image)).convert('RGBA')
        im = resize_img(im, size=size)

        # create writer
        writer = cv2.VideoWriter(output, cv2.VideoWriter_fourcc(*'avc1'), frame_rate, im.size)

        # create fixed poisson disk sampler as pixel grid
        COUNTER.set_verbose(False)
        w, h = im.size
        sampler = PoissonDiskSampler(width=w, height=h, radius=radius)
        BAR.set_verbose(True)
        BAR.reset('Rendering frames:', max=max_write_frames, item='frames')

        xy_offset = int((sampler.radius / np.sqrt(2)))
        sample_func = self.__get_sample_func(xy_offset=xy_offset)
        points = np.array(sampler.get_points()).astype('int64')

        read_count = -1
        write_count = -1
        success = True
        try:
            while True:

                # exit loop if max duration in frames is exceeded
                if write_count > max_write_frames:
                    break

                # decode frame
                success = self.reader.grab()
                if not success:
                    break

                # increase current input frame and check if we need to make mosaic
                read_count += 1

                # get output frame index
                output_index = int(read_count / max_read_frames * max_write_frames)

                # make mosaic if mosaic doesn't exist for output index
                if output_index > write_count:
                    success, frame = self.reader.retrieve()
                    if not success:
                        break
                    write_count += 1

                    self.__write_frame(writer=writer,
                                       reader_frame=frame,
                                       points=points,
                                       size=size,
                                       sample_func=sample_func)
                    BAR.next()

        except KeyboardInterrupt:
            pass
        BAR.finish()
        writer.release()

    def __get_sample_func(self, xy_offset: int) -> Callable:
        """ Creates function to run for every point in the Poisson sampler """

        def sample_func(input_frame: Image.Image, output_frame: Image.Image, point: Iterable) -> None:
            # get x, y coordinates
            x, y = point

            # get squared segment bounds from target
            left, top, right, bottom = np.array([x-xy_offset, y-xy_offset, x+xy_offset, y+xy_offset]).astype('int64')

            # get deterministically pseudo-random value from x, y coordinates
            theta_offset = int(xy_random(x, y) * 360)

            # extract target segment
            segment = input_frame.crop(box=((left, top, right, bottom)))

            # extract feature from segment
            feature = self.corpus.feature_extraction_func(segment)

            # get best match for feature
            image_index = self.corpus.tree.query([feature], k=1)[1][0][0]
            best_match = self.corpus.images[image_index]

            # apply transformation to match
            direction = [-1, 1][int(theta_offset) % 2 == 0]
            best_match = best_match.rotate(theta_offset + (self.theta*direction), resample=Image.Resampling.BICUBIC, expand=1)

            # paste transformed match in current frame
            box = tuple((point - np.array(best_match.size) // 2).astype('int64'))
            output_frame.paste(best_match, box=box, mask=best_match)

        return sample_func

    def __write_frame(self,
                      writer: cv2.VideoWriter,
                      reader_frame: np.ndarray,
                      points: Iterable,
                      size: Iterable,
                      sample_func: Callable) -> None:
        """ Creates mosaic for input frame """

        # prepare input and output
        input_frame = resize_img(Image.fromarray(bgr2rgb(reader_frame)).convert("RGBA"), size=size)
        output_frame = Image.new(mode='RGBA', size=input_frame.size, color=(0, 0, 0, 0))

        # create mosaic for current frame
        list(map(lambda sample: sample_func(input_frame, output_frame, sample), points))

        # write it into video
        write_frame(writer, output_frame)

        self.theta += 1
