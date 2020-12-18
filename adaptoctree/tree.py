"""
Construct an adaptive balanced linear octree form a set of points.
"""

import numba
import numpy as np

import adaptoctree.morton as morton


@numba.njit(parallel=True, cache=True)
def build(points, max_level=16, max_points=100, start_level=1):

    max_bound, min_bound = morton.find_bounds(points)
    x0 = morton.find_center(max_bound, min_bound)
    r0 = morton.find_radius(x0, max_bound, min_bound)

    keys = morton.encode_points_smt(
        points, start_level, x0, r0
    )
    unique_keys = np.unique(keys)
    n_unique_keys = len(unique_keys)

    for key_idx in numba.prange(n_unique_keys):

        todo_indices = np.where(keys == unique_keys[key_idx])[0]

        _build(
            points,
            max_level,
            max_points,
            x0,
            r0,
            keys,
            todo_indices,
            start_level,
        )

    return keys


@numba.njit(cache=True)
def _build(
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


@numba.njit(cache=True)
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


@numba.njit(cache=True)
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

    def bfs(root, tree, depth):
        """
        Perform breadth-first search starting from a given root, to find
            children in the tree that it overlaps with.

        Parameters:
        -----------
        root : np.int64
            Root of BFS.
        tree : {np.int64}
            Linear octree.
        depth : np.int64
            Maximum depth of octree.

        Returns:
        --------
        {np.int64}
            Set of overlapping children, if they exist.
        """
        queue = [root]

        overlaps = set()

        while queue:
            for node in queue:
                level = morton.find_level(node)
                new_queue = []
                for l in range(1, depth-level + 1):

                    descs = morton.find_descendents(node, l)
                    ints = set(descs).intersection(tree)

                    overlaps.update(ints)

                    new_queue.extend(list(ints))

            queue = new_queue

        return overlaps

    unique = set(tree)

    for node in tree:
        if bfs(node, unique, depth):
            unique.remove(node)

    return unique


@numba.njit(cache=True)
def balance_helper(tree, depth):
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
        Balanced linear octree, containing overlaps.
    """
    balanced = set(tree)

    for l in range(depth, 0, -1):
        # nodes at current level
        Q = [x for x in balanced if morton.find_level(x) == l]

        for q in Q:
            parent = morton.find_parent(q)
            neighbours = morton.find_neighbours(q)

            # Add neighbours, and neighbour siblings - may overlap
            siblings = set(morton.find_siblings(q))
            for n in neighbours:
                balanced.update(set(morton.find_siblings(n)))
                balanced.update(set([morton.find_parent(n)]))

    return numba.typed.List(balanced)


def balance(tree, depth):
    """
    Wrapper for JIT'd balance functions.

    Parameters:
    -----------
    tree : np.array(dtype=np.int64)
    depth : np.int64

    Returns:
    --------
    np.array(np.int64)
        Balanced octree
    """
    return  remove_overlaps(balance_helper(tree, depth), depth)


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
    leaves = -1*np.ones(n_points, dtype=np.int64)

    # Loop over points, and assign to a node from the tree by examining the bounds
    depth = find_depth(tree)

    for i, point in enumerate(points):
        for level in range(1, depth+1):
            key = morton.encode_point(point, level, x0, r0)
            if key in tree:
                leaves[i] = key

    return leaves


@numba.njit(cache=True)
def find_depth(tree):
    """
    Return maximum depth of a linear octree.

    Parameters:
    -----------
    tree : np.array(np.int64)

    Return:
    -------
    np.int64
    """
    return max(morton.find_level(np.unique(tree)))
