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


def balance(tree, depth):
    """
    Single-node sequential tree balancing. Based on Algorithm 8 in Sundar et al
        (2012).

    Parameters:
    -----------
    octree : Octree
    depth : int

    Returns:
    --------
    Octree
    """

    W = tree

    P = None
    balanced = None

    for level in range(depth, 0, -1):

        # Working list, filtered to current level
        Q = W[W[:,1] == level]

        # Q.sort()
        T = []
        len_Q = Q.shape[0]
        T_mask = np.zeros(len_Q, dtype=bool)

        parents = set()

        for i, q in enumerate(Q):
            parent = morton.find_parent(q[0])
            if parent not in parents:
                T_mask[i] = True
                parents.add(parent)

        T = Q[T_mask]

        for t in T:
            siblings = morton.find_siblings(t[0])
            neighbours = morton.find_neighbours(morton.find_parent(t[0]))

            sibling_levels = morton.find_level(siblings)
            neighbour_levels = morton.find_level(neighbours)

            tmp_siblings = np.c_[siblings, sibling_levels]
            tmp_neigbours = np.c_[neighbours, neighbour_levels]

            if balanced is None:
                balanced = tmp_siblings
            else:
                balanced = np.r_[balanced, tmp_siblings]

            if P is None:
                P = tmp_neigbours
            else:
                P = np.r_[P, tmp_neigbours]

        # Remove duplicates in P, if they exist
        P = np.r_[P, W[W[:,1]==(level-1)]]
        if P.shape[0] > 0:
            P = np.unique(P, axis=0)

        W = np.r_[W, P]
        P = None

    # Sort and linearise
    tmp = np.sort(balanced[:,0])
    linearised = linearise(tmp)
    levels = morton.find_level(linearised)
    return np.c_[linearised, levels]


def linearise(tree):
    """
    Remove overlaps in a sorted linear tree. Algorithm 7 in Sundar (2012).

    Parameters:
    -----------
    tree : np.array(dtype=np.int64)

    Returns:
    --------
    np.array(np.int64)
    """

    mask = np.zeros_like(tree, dtype=bool)

    n_octants = tree.shape[0]

    for i in range(n_octants):
        contained = False
        for j in range(n_octants):
            if i != j:
                # Check if ancestor contained in mask
                if not morton.not_ancestor(tree[i], tree[j]):
                    contained = True
                    mask[i] = True

            if contained:
                mask[i] = False
                break

        if not contained:
            mask[i] = True

    print(mask)
    return tree[mask]


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

        leaves[i] = tree[:, 0][leaf_index]

    return leaves
