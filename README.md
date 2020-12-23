<h1 align='center'> AdaptOctree </h1>

AdaptOctree is an library to **build** and **balance** adaptive linear octrees in Python, and Python's numeric ecosystem.

Adaptive linear octrees are a data structure useful in a large variety of scientific and numerical codes. AdaptOctree has been designed for use within [PyExaFMM](https://github.com/exafmm/pyexafmm), a Pythonic Kernel Independent Fast Multipole Method implementation. However, it is quite happy to work on its own too.

AdaptOctree is a work in process, please do report issues as you find them.

# Installation

Install from source into a Conda/Miniconda environment.

```bash
# Clone repository
git clone git@github.com:Excalibur-SLE/AdaptOctree.git
cd AdaptOctree

# Build Conda package
conda build conda.recipe

# Install conda package
conda install --use-local adaptoctree
```

# Usage

The code consists of two basic modules:

## 1) `adaptoctree.morton`

This module is used for generating Morton encodings and decodings for 3D Cartesian Points.

We provide a number of optimised methods for handling Morton encodings, for example neighbour searching, children/parent searching etc.


e.g. Generating the encodings for some points.

```python
import numpy as np
import adaptoctree.morton as morton

# Generate randomly distributed 3D points in a unit cube domain
points = np.random.rand(n_points, 3)

# Calculate implicit octree parameters
## Furthest corners of octree root node, with respect to origin
max_bound, min_bound = morton.find_bounds(points)

## Octree root node center
x0 = morton.find_center(max_bound, min_bound)

## Octree root node half side length
r0 = morton.find_radius(x0, max_bound, min_bound)

# Apply Morton encoding, at a given octree level
level = 1
keys = morton.encode_points(points, level, x0, r0)
```

## 2) `adaptoctree.tree`

This module is used for building and balancing trees, from a set of points.

e.g. Building and balancing.

```python
import numpy as np
import adaptoctree.tree as tree

# Generate randomly distributed 3D points in a unit cube domain
points = np.random.rand(n_points, 3)

# Build parameters
max_level = 16 # Maximum level allowed in AdaptOctree
max_points = 100 # The maximum points per node constraint
start_level = 1 # Initial level, to start tree construction from

unbalanced = tree.build(points, max_level, max_points, start_level)

# Now balance the unbalanced tree
depth = np.max(morton.find_level(unbalanced)) # Maximum depth achieved in octree

balanced = tree.balance(unbalanced, depth)
```

Note: The first import of AdaptOctree will generate a cache of Numba compiled functions, and therefore might take some time.

# Contributing

We welcome any contributions, check the open issues for currently troublesome bugs or feature requests.

We follow the same [developer guidelines](https://github.com/exafmm/pyexafmm/wiki/Contributing-%F0%9F%92%BB) as the PyExaFMM project.

# Citation

If you decide to cite our work, please do drop us an [email](mailto:srinathkailasa@gmail.com)!

We'd love to know what kind of projects you plan to use AdaptOctree in.

## References

[1] Sundar, H., Sampath, R. S., & Biros, G. (2008). Bottom-up construction and 2: 1 balance refinement of linear octrees in parallel. SIAM Journal on Scientific Computing, 30(5), 2675-2708.

[2] Tu, T., O'Hallaron, D. R., & Ghattas, O. (2005, November). Scalable parallel octree meshing for terascale applications. In SC'05: Proceedings of the 2005 ACM/IEEE conference on Supercomputing (pp. 4-4). IEEE.

[3] [The ExaFMM Project](https://github.com/exafmm)
