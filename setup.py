from setuptools import setup
import versioneer

requirements = [
    # package requirements go here
]

setup(
    name='AdaptOctree',
    version=versioneer.get_version(),
    cmdclass=versioneer.get_cmdclass(),
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
        'Programming Language :: Python :: 2.7',
        'Programming Language :: Python :: 3.6',
        'Programming Language :: Python :: 3.7',
    ]
)
