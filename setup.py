import os

from setuptools import setup, find_packages

HERE = os.path.abspath(os.path.dirname(__file__))
PATH_VERSION = os.path.join(HERE, 'adaptoctree', '__version.py')

ABOUT = {}

with open(PATH_VERSION, mode='r', encoding='utf-8') as f:
    exec(f.read(), ABOUT)

requirements = [
    "numba",
    "numpy",
    "pytest"
]


setup(
    name=ABOUT['__title__'],
    version=ABOUT['__version__'],
    description=ABOUT['__description__'],
    license="BSD",
    author="Srinath Kailasa",
    author_email='srinathkailasa@gmail.com',
    url='https://github.com/excalibur-sle/AdaptOctree',
    packages=find_packages(
        exclude=['*.test']
    ),
    entry_points={
        'console_scripts': [
            'adaptoctree=adaptoctree.cli:cli'
        ]
    },
    install_requires=requirements,
    keywords='AdaptOctree',
    classifiers=[
        'Programming Language :: Python :: 3.8',
    ]
)
