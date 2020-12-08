import itertools
import time

import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import numpy as np

from adaptoctree.tree import balance, build
import adaptoctree.morton as morton


def export_to_vtk(leafs, fname, octree_center, octree_radius):
    """Export tree to VTK."""
    from pyevtk.hl import unstructuredGridToVTK
    from pyevtk.vtk import VtkHexahedron

    nleafs = len(leafs)
    nvertices = nleafs * 8
    connectivity = np.arange(nvertices)
    vertices = np.empty((nvertices, 3), dtype=np.float64)
    cell_data = {"Level": np.empty(nleafs, dtype=np.int32)}
    offsets = 8 * np.arange(1, nleafs + 1)
    cell_types = VtkHexahedron.tid * np.ones(nleafs)

    for index, key in enumerate(leafs):
        level = morton.find_level(key)
        bounds = morton.find_node_bounds(key, octree_center, octree_radius)

        vertices[8 * index + 0, :] = np.array(
            [bounds[0, 0], bounds[0, 1], bounds[0, 2]]
        )
        vertices[8 * index + 1, :] = np.array(
            [bounds[1, 0], bounds[0, 1], bounds[0, 2]]
        )
        vertices[8 * index + 2, :] = np.array(
            [bounds[1, 0], bounds[1, 1], bounds[0, 2]]
        )
        vertices[8 * index + 3, :] = np.array(
            [bounds[0, 0], bounds[1, 1], bounds[0, 2]]
        )
        vertices[8 * index + 4, :] = np.array(
            [bounds[0, 0], bounds[0, 1], bounds[1, 2]]
        )
        vertices[8 * index + 5, :] = np.array(
            [bounds[1, 0], bounds[0, 1], bounds[1, 2]]
        )
        vertices[8 * index + 6, :] = np.array(
            [bounds[1, 0], bounds[1, 1], bounds[1, 2]]
        )
        vertices[8 * index + 7, :] = np.array(
            [bounds[0, 0], bounds[1, 1], bounds[1, 2]]
        )
        cell_data["Level"][index] = level

    unstructuredGridToVTK(
        fname,
        vertices[:, 0].copy(),
        vertices[:, 1].copy(),
        vertices[:, 2].copy(),
        connectivity=connectivity,
        offsets=offsets,
        cell_types=cell_types,
        cellData=cell_data,
    )


def plot_tree(octree, balanced, sources, octree_center, octree_radius):
    """

    Parameters:
    -----------
    octree : Octree
    """

    points = []

    fig1 = plt.figure()
    fig2 = plt.figure()
    ax1 = fig1.add_subplot(111, projection="3d")
    ax2 = fig2.add_subplot(111, projection="3d")

    unique = []

    for node in octree:
        level = morton.find_level(node)
        radius = octree_radius / (1 << level)

        center = morton.find_center_from_key(node, octree_center, octree_radius)

        r = [-radius, radius]

        for s, e in itertools.combinations(
            np.array(list(itertools.product(r, r, r))), 2
        ):
            if np.sum(np.abs(s - e)) == r[1] - r[0]:
                ax1.plot3D(*zip(s + center, e + center), color="b")

    for node in balanced:
        level = morton.find_level(node)
        radius = octree_radius / (1 << level)

        center = morton.find_center_from_key(node, octree_center, octree_radius)

        r = [-radius, radius]

        for s, e in itertools.combinations(
            np.array(list(itertools.product(r, r, r))), 2
        ):
            if np.sum(np.abs(s - e)) == r[1] - r[0]:
                ax2.plot3D(*zip(s + center, e + center), color="b")

    # Plot particle data
    ax1.scatter(sources[:, 0], sources[:, 1], sources[:, 2], c="g", s=0.8)
    ax2.scatter(sources[:, 0], sources[:, 1], sources[:, 2], c="g", s=0.8)
    plt.show()


def make_spiral(N):

    theta = np.linspace(0, 2 * np.pi, N)
    phi = np.linspace(0, np.pi, N)

    x = np.cos(theta) * np.sin(phi)
    y = np.sin(theta) * np.sin(phi)
    z = np.cos(phi)

    return np.c_[x, y, z]


def make_moon(npoints):

    x = np.linspace(0, 2 * np.pi, npoints) + np.random.rand(npoints)
    y = 0.5 * np.ones(npoints) + np.random.rand(npoints)
    z = np.sin(x) + np.random.rand(npoints)

    moon = np.array([x, y, z]).T
    return moon


def main():
    np.random.seed(0)

    N = int(1000)
    particles = make_moon(N)
    # sources = targets = np.random.rand(N, 3)
    # sources = targets = make_spiral(N)

    maximum_level = 10
    max_num_particles = 70
    max_bound, min_bound = morton.find_bounds(particles)

    octree_center = morton.find_center(max_bound, min_bound)
    octree_radius = morton.find_radius(octree_center, max_bound, min_bound)

    start = time.time()
    unbalanced = build(particles, max_num_particles=max_num_particles, maximum_level=maximum_level)
    unbalanced = np.unique(unbalanced)
    depth = max(morton.find_level(unbalanced))

    balanced = balance(unbalanced, depth, maximum_level)

    plot_tree(unbalanced, balanced, particles, octree_center, octree_radius)

    # print(balanced)
    # print()
    # print(unbalanced)


if __name__ == "__main__":
    main()