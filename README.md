<h1 align='center'> AdaptOctree </h1>

AdaptOctree is an implementation of Adaptive Linear Octrees in Python, and it's numeric ecosystem.

AdaptOctree has been designed for use within PyExaFMM, a Pythonic Kernel Independent Fast Multipole Method
implementation. However, it is quite happy to work on its own too.

# Installation

Install from source into a Conda/Miniconda environment.

```bash
# Clone repository
git clone git@github.com:Excalibur-SLE/AdaptOctree.git

# Build Conda package
conda build conda.recipe

# Install conda package
conda install --use-local adaptoctree
```

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
