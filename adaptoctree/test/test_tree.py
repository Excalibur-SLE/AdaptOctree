"""
Test tree construction.


Need some kind of test to ensure that all points are assigned. There's a
relationship between maximum level and max number of particles per box that needs
to observed.
"""
import itertools
import time

import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import numpy as np

from adaptoctree.tree import Octree, linearise, balance
import adaptoctree.morton as morton


def plot_tree(octree, balanced, sources, octree_center, octree_radius):
    """

    Parameters:
    -----------
    octree : Octree
    """

    points = []

    fig1 = plt.figure()
    fig2 = plt.figure()
    ax1 = fig1.add_subplot(111, projection='3d')
    ax2 = fig2.add_subplot(111, projection='3d')

    unique = []

    for node in octree:
        level = morton.find_level(node)
        radius = octree_radius / (1 << level)

        center = morton.find_center_from_key(node, octree_center, octree_radius)

        r = [-radius, radius]

        for s, e in itertools.combinations(np.array(list(itertools.product(r, r, r))), 2):
            if np.sum(np.abs(s-e)) == r[1]-r[0]:
                ax1.plot3D(*zip(s+center, e+center), color="b")

    for node in balanced:
        level = morton.find_level(node)
        radius = octree_radius / (1 << level)

        center = morton.find_center_from_key(node, octree_center, octree_radius)

        r = [-radius, radius]

        for s, e in itertools.combinations(np.array(list(itertools.product(r, r, r))), 2):
            if np.sum(np.abs(s-e)) == r[1]-r[0]:
                ax2.plot3D(*zip(s+center, e+center), color="b")

    # Plot particle data
    ax1.scatter(sources[:, 0], sources[:, 1], sources[:, 2], c='g', s=0.8)
    ax2.scatter(sources[:, 0], sources[:, 1], sources[:, 2], c='g', s=0.8)
    plt.show()


def make_moon(npoints):

    x = np.linspace(0, 2*np.pi, npoints) + np.random.rand(npoints)
    y = 0.5*np.ones(npoints) + np.random.rand(npoints)
    z = np.sin(x) + np.random.rand(npoints)

    moon = np.array([x, y, z]).T
    return moon


def main():
    np.random.seed(0)

    N = int(500)
    sources = targets = make_moon(N)
    # sources = targets = np.random.rand(N, 3)

    tree_conf = {
        "sources": sources,
        "targets": targets,
        "maximum_level": 10,
        "maximum_particles": 10
    }

    # Sort sources and targets by octant at level 1 of octree
    start = time.time()
    octree = Octree(**tree_conf)
    print(f"initial run: {time.time() - start}")

    max_bound, min_bound = morton.find_bounds(tree_conf['sources'], tree_conf['targets'])
    octree_center = morton.find_center(max_bound, min_bound)
    octree_radius = morton.find_radius(octree_center, max_bound, min_bound)

    balanced = balance(octree)

    original = octree.tree
    plot_tree(original, balanced, octree.sources, octree_center, octree_radius)

    # print(octree.level_index_pointer)
    # print([morton.find_level(o) for o in octree.tree])
    # print(octree.depth)

if __name__ == "__main__":
    main()