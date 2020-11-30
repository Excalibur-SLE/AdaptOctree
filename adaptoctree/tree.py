"""
Construct an adaptive linear octree form a set of points.
"""

import numba
import numpy as np

import adaptoctree.morton as morton


@numba.njit(parallel=True)
def build(particles, maximum_level=16, max_num_particles=100, first_level=1):

    max_bound, min_bound = morton.find_bounds(particles)
    octree_center = morton.find_center(max_bound, min_bound)
    octree_radius = morton.find_radius(octree_center, max_bound, min_bound)

    morton_keys = morton.encode_points_smt(
        particles, first_level, octree_center, octree_radius
    )
    unique_indices = np.unique(morton_keys)
    n_unique_indices = len(unique_indices)
    for index in numba.prange(n_unique_indices):
        todo_indices = np.where(morton_keys == unique_indices[index])[0]
        build_implementation(
            particles,
            maximum_level,
            max_num_particles,
            octree_center,
            octree_radius,
            morton_keys,
            todo_indices,
            first_level,
        )
    return morton_keys


@numba.njit
def build_implementation(
    particles,
    maximum_level,
    max_num_particles,
    octree_center,
    octree_radius,
    morton_keys,
    todo_indices,
    first_level,
):

    level = first_level

    todo_indices_sorted = todo_indices[np.argsort(morton_keys[todo_indices])]

    while True:
        if level == (maximum_level):
            break
        todo_list = process_level(todo_indices_sorted, morton_keys, max_num_particles)
        ntodo = len(todo_list)
        if ntodo == 0:
            break
        todo_indices = np.empty(ntodo, dtype=np.int64)
        for index in range(ntodo):
            todo_indices[index] = todo_list[index]
        if len(todo_indices) == 0:
            # We are done
            break
        else:
            morton_keys[todo_indices] = morton.encode_points(
                particles[todo_indices], level + 1, octree_center, octree_radius
            )
            todo_indices_sorted = todo_indices[np.argsort(morton_keys[todo_indices])]
            level += 1
    return morton_keys


@numba.njit
def process_level(sorted_indices, morton_keys, max_num_particles):
    """Process a level."""
    count = 0
    pivot = morton_keys[sorted_indices[0]]
    nindices = len(sorted_indices)
    todo = numba.typed.List.empty_list(numba.types.int64, allocated=nindices)
    trial_set = numba.typed.List.empty_list(numba.types.int64, allocated=nindices)
    for index in range(nindices):
        if morton_keys[sorted_indices[index]] != pivot:
            if count > max_num_particles:
                todo.extend(trial_set)
            trial_set.clear()
            pivot = morton_keys[sorted_indices[index]]
            count = 0
        count += 1
        trial_set.append(sorted_indices[index])
    # The last element in the for-loop might have
    # a too large count. Need to process this as well
    if count > max_num_particles:
        todo.extend(trial_set)
    return todo


def balance(unbalanced_tree, depth, max_level):
    """
    Single-node sequential tree balancing.

    Parameters:
    -----------
    octree : Octree
    depth : int

    Returns:
    --------
    Octree
    """
    from adaptoctree.morton import find_level

    work_set = unbalanced_tree.copy()

    balanced = None
    for level in range(depth, 0, -1):

        # Working list, filtered to current level
        filter_fun = lambda elem: find_level(elem) == level
        work_subset = filter(filter_fun, work_set)
        # work_subset = work_set[work_set[:, 1] == level]

        # Find if neighbours of leaves at this level violate balance constraint

        for key in work_subset:
            neighbours = morton.find_neighbours(key)
            n_neighbours = len(neighbours)

            # Invalid neighbours are any that are more than two levels coarse than the current level

            n_invalid_neighbours = n_neighbours * (level - 2)
            invalid_neighbours = np.empty(shape=(n_invalid_neighbours), dtype=np.int64)

            i = 0
            for neighbour in neighbours:
                for invalid_level in range(level - 2, 0, -1):

                    # remove level bits
                    invalid_neighbour = neighbour >> 15
                    # add bits for invalid level key
                    invalid_neighbour = invalid_neighbour >> (
                        3 * (level - invalid_level)
                    )

                    # add new level bits
                    invalid_neighbour = invalid_neighbour << 15
                    invalid_neighbour = invalid_neighbour | invalid_level
                    invalid_neighbours[i] = invalid_neighbour
                    i += 1

            invalid_neighbours = np.unique(invalid_neighbours)
            found, invalid_neighbours_idx, W_idx = np.intersect1d(
                invalid_neighbours, work_set, return_indices=True
            )

            # Check if invalid neighbours exist in working list for this node,
            # q, if so remove them and replace with valid descendents
            # Within 1 level of coarseness of q
            if found.size > 0:
                for invalid_neighbour in invalid_neighbours:
                    invalid_level = morton.find_level(invalid_neighbour)
                    valid_children = morton.find_descendents(
                        invalid_neighbour, invalid_level - (level + 1)
                    )

                    #  Filter out from W
                    work_set = work_set[work_set != invalid_neighbour]

                    # Add valid descendents to W
                    work_set = np.r_[work_set, valid_children]

        if balanced is None:
            balanced = work_subset

        else:
            balanced = np.r_[balanced, work_subset]

    return np.unique(balanced, axis=0)


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
        lower_bounds[i : i + 2, :] = bounds[0, :]
        upper_bounds[i : i + 2, :] = bounds[1, :]

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
