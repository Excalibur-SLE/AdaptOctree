"""
Test tree construction
"""
import itertools
import time

import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import numpy as np

from adaptoctree.tree import Octree
import adaptoctree.morton as morton


def plot_tree(tree, octree_center, octree_radius):
    """

    Parameters:
    -----------
    tree : [Node]
    """

    points = []

    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')

    unique = []

    for node in tree:
        level = morton.find_level(node.key)
        radius = octree_radius / (1 << level)

        center = morton.find_center_from_key(node.key, octree_center, octree_radius)

        r = [-radius, radius]

        for s, e in itertools.combinations(np.array(list(itertools.product(r, r, r))), 2):
            if np.sum(np.abs(s-e)) == r[1]-r[0]:
                ax.plot3D(*zip(s+center, e+center), color="b")

    # Plot particle data
    sources = tree[0].sources
    ax.scatter(sources[:, 0], sources[:, 1], sources[:, 2], c='g', s=0.8)
    plt.show()


def make_moon(npoints):

    x = np.linspace(0, 2*np.pi, npoints) + np.random.rand(npoints)
    y = 0.5*np.ones(npoints) + np.random.rand(npoints)
    z = np.sin(x) + np.random.rand(npoints)

    moon = np.array([x, y, z]).T
    return moon


def main():
    np.random.seed(0)

    N = int(1e4)
    # sources = targets = make_moon(N)
    sources = targets = np.random.rand(N, 3)

    tree_conf = {
        "sources": sources,
        "targets": targets,
        "maximum_level": 15,
        "maximum_particles": 5
    }

    # Sort sources and targets by octant at level 1 of octree
    start = time.time()
    octree = Octree(**tree_conf)
    print(f"initial run: {time.time() - start}")

    max_bound, min_bound = morton.find_bounds(tree_conf['sources'], tree_conf['targets'])
    octree_center = morton.find_center(max_bound, min_bound)
    octree_radius = morton.find_radius(octree_center, max_bound, min_bound)

    # plot_tree(octree.tree, octree_center, octree_radius)

    print(f"number of octants in tree {octree.size}")


if __name__ == "__main__":
    main()