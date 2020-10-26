"""
Construct an adaptive linear octree form a set of points.
"""
import numba
import numpy as np

import morton


class Node:
    """
    Minimal octree node.
    """
    def __init__(self, key, sources, targets, children=None):
        self.key = key
        self.sources = sources
        self.targets = targets
        if children is not None:
            self.children = children

    def __repr__(self):
        return f"<morton_id>{self.key}</morton_id>"


def build_tree(
    sources,
    targets,
    maximum_level,
    maximum_particles,
    ):
    """
    Top-down construction of an adaptive octre mesh.

    Parameters:
    -----------
    sources : np.array(shape=(nsources, 3))
    targets : np.array(shape=(nsources, 3))
    maximum_level : np.int32
        Maximum level of the octree.
    maximum_particles : np.int32
        Maximum number of particles per node.

    Returns:
    --------
    [Node]
        Adaptive linear Octree.
    """

    max_bound, min_bound = morton.find_bounds(sources, targets)
    octree_center = morton.find_center(max_bound, min_bound)
    octree_radius = morton.find_radius(octree_center, max_bound, min_bound)

    tree = [Node(0, sources, targets)]
    built = False
    level = 0

    while not built:

        if (level == maximum_level):
            built = True

        source_keys = morton.encode_points(sources, level, octree_center, octree_radius)
        target_keys = morton.encode_points(targets, level, octree_center, octree_radius)

        particle_keys = np.hstack((source_keys, target_keys))
        particle_index_array = np.argsort(particle_keys)

        unique_keys, counts = np.unique(particle_keys, return_counts=True) # O(N)

        refined_sources = []
        refined_targets = []

        for i, count in enumerate(counts):
            leaf = unique_keys[i]
            if count > maximum_particles:
                source_idxs = np.where(source_keys == leaf)
                target_idxs = np.where(target_keys == leaf)
                refined_sources.append(sources[source_idxs])
                refined_targets.append(targets[target_idxs])

            else:
                source_idxs = np.where(source_keys == leaf)
                target_idxs = np.where(target_keys == leaf)

                tree.append(
                    Node(
                        key=unique_keys[i],
                        sources=sources[source_idxs],
                        targets=targets[target_idxs]
                        )
                    )

        level += 1

        if (not refined_sources) or (not refined_targets):
            built = True

        else:
            sources = np.concatenate(refined_sources)
            targets = np.concatenate(refined_targets)

    return tree


def plot_tree(tree, octree_center, octree_radius):
    """

    Parameters:
    -----------
    tree : [Node]
    """

    import itertools

    import matplotlib.pyplot as plt
    from mpl_toolkits.mplot3d import Axes3D

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


def main():
    np.random.seed(0)

    def make_moon(npoints):

        x = np.linspace(0, 2*np.pi, npoints) + np.random.rand(npoints)
        y = 0.5*np.ones(npoints) + np.random.rand(npoints)
        z = np.sin(x) + np.random.rand(npoints)

        moon = np.array([x, y, z]).T
        return moon

    sources = targets = make_moon(250)

    tree_conf = {
        "sources": sources,
        "targets": targets,
        "maximum_level": 5,
        "maximum_particles": 50
    }

    # Sort sources and targets by octant at level 1 of octree
    tree = build_tree(**tree_conf)

    max_bound, min_bound = morton.find_bounds(tree_conf['sources'], tree_conf['targets'])
    octree_center = morton.find_center(max_bound, min_bound)
    octree_radius = morton.find_radius(octree_center, max_bound, min_bound)
    plot_tree(tree, octree_center, octree_radius)


if __name__ == "__main__":
    main()
