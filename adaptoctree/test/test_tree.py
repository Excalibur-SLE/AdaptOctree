"""
Test tree construction
"""
import itertools
import time

import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import numpy as np

from adaptoctree.tree import Octree, linearise
import adaptoctree.morton as morton


def plot_tree(octree, octree_center, octree_radius):
    """

    Parameters:
    -----------
    octree : Octree
    """

    points = []

    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')

    unique = []

    for node in octree.tree:
        level = morton.find_level(node.key)
        radius = octree_radius / (1 << level)

        center = morton.find_center_from_key(node.key, octree_center, octree_radius)

        r = [-radius, radius]

        for s, e in itertools.combinations(np.array(list(itertools.product(r, r, r))), 2):
            if np.sum(np.abs(s-e)) == r[1]-r[0]:
                ax.plot3D(*zip(s+center, e+center), color="b")

    # Plot particle data
    sources = octree.sources
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

    N = int(10)
    # sources = targets = make_moon(N)
    sources = targets = np.random.rand(N, 3)

    tree_conf = {
        "sources": sources,
        "targets": targets,
        "maximum_level": 15,
        "maximum_particles": 25
    }

    # Sort sources and targets by octant at level 1 of octree
    start = time.time()
    octree = Octree(**tree_conf)
    print(f"initial run: {time.time() - start}")

    max_bound, min_bound = morton.find_bounds(tree_conf['sources'], tree_conf['targets'])
    octree_center = morton.find_center(max_bound, min_bound)
    octree_radius = morton.find_radius(octree_center, max_bound, min_bound)

    linearise(octree)

    plot_tree(octree, octree_center, octree_radius)

    # Furthest corner
    # anchor = [0, 0, 0, 1]
    # desc = [1, 1, 1, 2]

    # maximum_level = 10

    # a = morton.encode_anchor(anchor)
    # b = morton.encode_anchor(desc)

    # print(f'a {bin(a)}')
    # print(f'b {bin(b)}')

    # print(f'not ancestor: {morton.not_ancestor(a, b)}')




if __name__ == "__main__":
    main()