from setuptools import setup, find_packages
from pathlib import Path

long_description = (Path(__file__).parent / "README.md").read_text()

MODULE = 'articon'
VERSION = '0.0.1'
DESCRIPTION = 'Corpus-based, icon mosaicking art in Python'

setup(
    name=MODULE,
    version=VERSION,
    author='Felipe Tovar-Henao',
    author_email='<felipe.tovar.henao@gmail.com>',
    description=DESCRIPTION,
    url='https://github.com/felipetovarhenao/articon',
    packages=find_packages(exclude=("tests",)),
    license='OSI Approved :: ISC License (ISCL)',
    install_requires=[
        'numpy',
        'typing_extensions',
        'Pillow',
        'opencv-python',
        'scikit-learn',
        'progress'
    ],
    keywords=['mosaicking', 'emoji art', 'machine learning', 'yung jake'],
    classifiers=[
        'Development Status :: 5 - Production/Stable',
        'Intended Audience :: Other Audience',
        'Natural Language :: English',
        'Operating System :: MacOS :: MacOS X',
        'Operating System :: Microsoft :: Windows',
        'Programming Language :: Python'
    ],
    long_description=long_description,
    long_description_content_type='text/markdown'
)
