# **articon**: _icon mosaicking art in python_
-----------------------

![version](https://img.shields.io/pypi/v/articon)
![downloads](https://img.shields.io/pypi/dm/articon)
![build](https://img.shields.io/github/actions/workflow/status/felipetovarhenao/articon/test.yaml?label=test)
![last_commit](https://img.shields.io/github/last-commit/felipetovarhenao/articon)
![license](https://img.shields.io/pypi/l/articon)

## Description
**articon** is a Python package for flexible, corpus-based icon mosaicking. Given a target image or video, **articon** tries to find the best matches from the corpus and assembled them into a mosaic.
Here are some emoji-art examples made with **articon**.


<div align="center">
    <img src="examples/starry-night-example.gif" alt="starry night" height="400px" width="auto" />
    <img src="examples/bob-ross-example.gif" alt="Bob Ross" height="400px" width="auto" />
    <img src="examples/trump-example.gif" alt="Trump" height="400px" width="auto" />
    <img src="examples/mona-lisa-example.gif" alt="Mona Lisa" height="400px" width="auto" />
</div>

To see a demo of video mosaics, click [here](https://youtu.be/K_I0N-L-HzU)

## Basic example

```python
from articon.models import IconCorpus, IconMosaic

source = 'path/to/icon/images/folder/'
target = 'path/to/target/image/'

# create corpus, resizing all images to fit within 40x40 pixels
corpus = IconCorpus.read(source=source, size=40)

# visualize corpus
corpus.show()

# create mosaic, pre-resizing target to fit within 900x900 pixels
mosaic = IconMosaic(target=target, 
                    corpus=corpus,
                    size=900)
# show mosaic
mosaic.show()

# write mosaic to disk
mosaic.save('mymosaic.png')

```


## Acknowledgements
This code is a Python porting and expansion of [emoji-mosaic](https://github.com/ericandrewlewis/emoji-mosaic) by [ericandrewlewis](https://github.com/ericandrewlewis/), and inspired by [Yung Jake](https://en.wikipedia.org/wiki/Yung_Jake)'s [emoji art](https://www.nytimes.com/2017/07/26/style/emoji-portraits-yung-jake.html).

## Installation
To install the latest version of **articon** with pip, run:
```shell
pip install articon
```

## Datasets
A convenient corpus dataset that works well with **articon** is [Kaggle](https://www.kaggle.com)'s *Full Emoji Image Dataset*, which you can download [here](https://www.kaggle.com/datasets/subinium/emojiimage-dataset?resource=download).


## License
ISC License
Copyright (c) 2023, Felipe Tovar-Henao

Permission to use, copy, modify, and/or distribute this software for any purpose with or without fee is hereby granted, provided that the above copyright notice and this permission notice appear in all copies.

THE SOFTWARE IS PROVIDED "AS IS" AND THE AUTHOR DISCLAIMS ALL WARRANTIES WITH REGARD TO THIS SOFTWARE INCLUDING ALL IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS. IN NO EVENT SHALL THE AUTHOR BE LIABLE FOR ANY SPECIAL, DIRECT, INDIRECT, OR CONSEQUENTIAL DAMAGES OR ANY DAMAGES WHATSOEVER RESULTING FROM LOSS OF USE, DATA OR PROFITS, WHETHER IN AN ACTION OF CONTRACT, NEGLIGENCE OR OTHER TORTIOUS ACTION, ARISING OUT OF OR IN CONNECTION WITH THE USE OR PERFORMANCE OF THIS SOFTWARE.