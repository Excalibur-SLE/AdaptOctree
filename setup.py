from setuptools import setup

requirements = [
    'numba',
    'numpy',
]

setup(
    name='AdaptOctree',
    version='0.0.0',
    description="Parallel adaptive octrees in Python",
    license="BSD",
    author="Srinath Kailasa",
    author_email='srinath.kailasa.18@ucl.ac.uk',
    url='https://github.com/excalibur-sle/AdaptOctree',
    packages=['adaptoctree'],
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
