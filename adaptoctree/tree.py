"""
Construct an adaptive linear octree form a set of points.
"""

import numba
import numpy as np

import adaptoctree.morton as morton


def build(
    sources,
    targets,
    maximum_level,
    maximum_particles,
    ):
    """
    Top-down construction of an adaptive octree mesh.

    Parameters:
    -----------
    sources : np.array(shape=(nsources, 3), dtype=np.float32)
    targets : np.array(shape=(nsources, 3), dtype=np.float32)
    maximum_level : np.int32
        Maximum level of the octree.
    maximum_particles : np.int32
        Maximum number of particles per node.

    Returns:
    --------
    Octree
        Unbalanced adaptive linear Octree.
    """

    max_bound, min_bound = morton.find_bounds(sources, targets)
    octree_center = morton.find_center(max_bound, min_bound)
    octree_radius = morton.find_radius(octree_center, max_bound, min_bound)

    tree = []

    built = False
    level = 1
    size = 1

    leaf_index = 0

    while not built:

        if (level == (maximum_level)):
            depth = maximum_level
            built = True

        # Heavy lifting
        source_keys = morton.encode_points(sources, level, octree_center, octree_radius)
        target_keys = morton.encode_points(targets, level, octree_center, octree_radius)

        particle_keys = np.hstack((source_keys, target_keys))

        particle_index_array = np.argsort(particle_keys)
        unique_keys, counts = np.unique(particle_keys, return_counts=True) # O(N)

        refined_sources = []
        refined_targets = []

        for i, count in enumerate(counts):
            leaf = unique_keys[i]

            source_idxs = np.where(source_keys == leaf)
            target_idxs = np.where(target_keys == leaf)

            if count > maximum_particles:
                refined_sources.append(sources[source_idxs])
                refined_targets.append(targets[target_idxs])

            else:
                leaf_index += 1
                tree.append((leaf, level))
                size += 1

        if (not refined_sources) or (not refined_targets):
            depth = level
            built = True

        else:
            sources = np.concatenate(refined_sources)
            targets = np.concatenate(refined_targets)

        level += 1

    return np.array(tree, dtype=np.int64), depth, size


def balance(unbalanced):

    pass


def assign_points_to_keys(points, tree, x0, r0):
    """
    Assign particle positions to Morton keys in a given tree.

    Parameters:
    -----------
    points : np.array(shape=(N, 3), dtype=np.float32)
    tree : Octree

    Returns:
    --------
    np.array(shape=(N,), dtype=np.int64)
        Column vector specifying the Morton key of the node that each point is
        associated with.
    """
    # Map Morton key to bounds that they represent.
    n_points = points.shape[0]
    n_keys = tree.shape[0]
    lower_bounds = np.zeros(shape=(n_keys, 3), dtype=np.float32)
    upper_bounds = np.zeros(shape=(n_keys, 3), dtype=np.float32)

    leaves = np.zeros(n_points, dtype=np.int64)

    # Loop over all nodes to find bounds
    for i in range(n_keys):
        key = tree[i, 0]
        bounds = morton.find_node_bounds(key, x0, r0)
        lower_bounds[i:i+2, :] = bounds[0, :]
        upper_bounds[i: i+2, :] = bounds[1, :]

    # Loop over points, and assign to a node from the tree by examining the bounds
    for i, point in enumerate(points):
        upper_bound_index = np.all(point < upper_bounds, axis=1)
        lower_bound_index = np.all(point >= lower_bounds, axis=1)
        leaf_index = upper_bound_index & lower_bound_index

        if np.count_nonzero(leaf_index) != 1:
            print(np.count_nonzero(leaf_index))
            print(point)
            print(tree[:, 0][leaf_index])
            a, b = tree[:, 0][leaf_index]
            print([morton.find_node_bounds(k, x0, r0) for k in tree[:, 0][leaf_index]])
            print(morton.not_ancestor(a, b))
            print()

        leaves[i] = tree[:, 0][leaf_index]

    return leaves
