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


def bfs(root, tree, depth):
    """
    Perform breadth-first search starting from a given root, to find
        children in the tree that it overlaps with.

    Parameters:
    -----------
    root : np.int64
        Root of BFS.
    tree : np.array(dtype=np.int64)
        Linear octree.
    depth : np.int64
        Maximum depth of octree.

    Returns:
    --------
    {np.int64}
        Set of overlapping children, if they exist.
    """
    tree = set(tree)
    queue = [root]

    overlaps = set()

    while queue:
        for node in queue:
            level = morton.find_level(node)
            new_queue = []
            for l in range(1, depth-level + 1):

                descs = set(morton.find_descendents(node, l))

                ints = descs.intersection(tree)
                overlaps.update(ints)
                new_queue.extend(list(ints))

        queue = new_queue

    return overlaps


def remove_overlaps(tree, depth):
    """
    Perform BFS, for each node in the linear octree, and remove if
        it overlaps with any present descendents in the tree.

    Parameters:
    -----------
    tree : np.array(dtype=np.int64)
    depth : np.int64

    Returns:
    --------
    {np.int64}
    """

    unique = set(tree)

    for node in tree:
        if bfs(node, unique, depth):
            unique.remove(node)

    return unique


def balance(tree, depth):
    """
    Perform balancing to enforece the 2:1 constraint between neighbouring
        nodes in linear octree.

    Parameters:
    -----------
    tree : np.array(dtype=np.int64)
    depth : np.int64

    Returns:
    --------
    {np.int64}
        Balanced octree
    """
    balanced = set(tree)

    for l in range(depth, 0, -1):
        # nodes at current level
        Q = {x for x in balanced if morton.find_level(x) == l}

        for q in Q:
            parent = morton.find_parent(q)
            neighbours = set(morton.find_neighbours(q))
            parent_neighbours = set(morton.find_neighbours(parent))
            balanced.update(parent_neighbours)
            balanced.update(neighbours)
    return remove_overlaps(list(balanced), depth)


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
    n_keys = len(tree)
    leaves = np.zeros(n_points, dtype=np.int64)

    # Loop over points, and assign to a node from the tree by examining the bounds
    for i, point in enumerate(points):
        for key in tree:
            lower_bound, upper_bound = morton.find_node_bounds(key, x0, r0)
            if (np.all(lower_bound <= point)) and (np.all(point < upper_bound )):
                leaves[i] = key

    return leaves