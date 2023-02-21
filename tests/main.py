import unittest
from imp.utils import create_image_palette
from imp.features import IconCorpus, IconMosaic
from imp.image import Image, ImageDraw, ImageStat
from collections.abc import Iterable
import os


class ImpTest(unittest.TestCase):

    def setUp(self):
        # create test directory
        self.test_dir = '.imp_test'
        if not os.path.exists(self.test_dir):
            os.mkdir(self.test_dir)

        data_dir = 'tests/data/'

        self.target_1 = os.path.join(data_dir, 'targets/mona-lisa.jpeg')
        self.target_2 = os.path.join(data_dir, 'targets/starry-night.jpeg')
        self.sources = os.path.join(data_dir, 'sources')

        # create test corpora
        self.color_corpus = IconCorpus(images=create_image_palette())
        self.arc_corpus = IconCorpus(images=create_image_palette(func=self.ring_from_color))
        self.emoji_corpus = IconCorpus.read(source=self.sources, size=(60, 60))

    def test_basic_mosaic(self):
        IconMosaic(target=self.target_1, corpus=self.emoji_corpus, radius=15)

    def test_custom_feature_extraction_func(self):
        def func(img):
            return ImageStat.Stat(img).mean[:3]
        c = IconCorpus.read(source=self.sources, feature_extraction_func=func)
        IconMosaic(target=self.target_2, corpus=c, scale_target=0.7)

    def tearDown(self) -> None:
        os.rmdir(self.test_dir)
        return super().tearDown()

    @staticmethod
    def ring_from_color(color: Iterable) -> Image.Image:
        im = Image.new(mode='RGBA', size=(60, 60), color=(0, 0, 0, 0))
        draw = ImageDraw.Draw(im, mode='RGBA')
        draw.ellipse(xy=(0, 0, *im.size), fill=(0, 0, 0, 0), outline=(*color, 255), width=20)
        return im
