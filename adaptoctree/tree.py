"""
Construct an adaptive balanced linear octree form a set of points.
"""

import numba
import numpy as np

import adaptoctree.morton as morton
import adaptoctree.types as types


@numba.njit(
    [types.LongIntList(types.LongArray, types.Keys, types.Int)],
    cache=True
)
def find_work_items(sorted_work_indices, keys, max_points):
    """
    Process a level, to find the work items corresponding to particles that
        occupy a certain node that exceed the maximum points per node threshold.
        This method returns the global indices of these particles.
    """
    count = 0
    pivot = keys[sorted_work_indices[0]]
    nindices = len(sorted_work_indices)

    todo = numba.typed.List.empty_list(numba.types.int64, allocated=nindices)
    trial_set = numba.typed.List.empty_list(numba.types.int64, allocated=nindices)

    for index in range(nindices):
        if keys[sorted_work_indices[index]] != pivot:
            if count > max_points:
                todo.extend(trial_set)
            trial_set.clear()
            pivot = keys[sorted_work_indices[index]]
            count = 0
        count += 1
        trial_set.append(sorted_work_indices[index])

    # The last element in the for-loop might have
    # a too large count. Need to process this as well
    if count > max_points:
        todo.extend(trial_set)

    return todo


@numba.njit(
    [
        types.Void(
            types.Coords, types.Int, types.Int, types.Coord,
            types.Float, types.Keys, types.LongArray, types.Int
        )
    ],
    cache=True
)
def build_helper(
    points, max_level, max_points, x0, r0, keys, work_indices, start_level
):
    """
    Build helper function. Works in-place on batches of points with same Morton
        key at the coarsest level 'start_level', and maintaining the max
        particles per node constraint.

        Strategy: For all particles in a given octant of the root node at the
        'start_level', calculate if any child octants violate the max_particles
        constraint - if they do, then repartition the particles into the
        grand child octants, and so on, until the constraint is satisfied.
    """

    level = start_level

    while True:
        if level == max_level:
            break

        work_list = find_work_items(work_indices, keys, max_points)

        ntodo = len(work_list)

        if ntodo == 0:
            break

        work_indices = np.empty(ntodo, dtype=np.int64)

        for index in range(ntodo):
            work_indices[index] = work_list[index]

        keys[work_indices] = morton.encode_points(
            points[work_indices], level + 1, x0, r0
        )

        work_indices = work_indices[np.argsort(keys[work_indices])]

        level += 1


@numba.njit(
    [types.Keys(types.Coords, types.Int, types.Int, types.Int)],
    parallel=True, cache=True
)
def build(points, max_level, max_points, start_level):
    """
    Build an unbalanced linear Morton encoded octree, that satisfied a the
        constraint of at most 'max_points' points per node.
    """

    max_bound, min_bound = morton.find_bounds(points)
    x0 = morton.find_center(max_bound, min_bound)
    r0 = morton.find_radius(x0, max_bound, min_bound)

    keys = morton.encode_points_smt(points, start_level, x0, r0)
    unique_keys = np.unique(keys)
    n_unique_keys = len(unique_keys)

    for key_idx in numba.prange(n_unique_keys):

        work_indices = np.where(keys == unique_keys[key_idx])[0]

        build_helper(
            points=points,
            max_level=max_level,
            max_points=max_points,
            x0=x0,
            r0=r0,
            keys=keys,
            work_indices=work_indices,
            start_level=start_level,
        )

    return keys


@numba.njit(
    [types.KeySet(types.KeyList, types.Int)],
    cache=True
)
def remove_overlaps(tree, depth):
    """
    Perform BFS, for each node in the linear octree, and remove if
        it overlaps with any present descendents in the tree.
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


@numba.njit(
    [types.KeyList(types.Keys, types.Int)],
    cache=True
)
def balance_helper(tree, depth):
    """
    Perform balancing to enforece the 2:1 constraint between neighbouring
        nodes in linear octree.
    """
    balanced = set(tree)

    for l in range(depth, 0, -1):
        # nodes at current level
        Q = [x for x in balanced if morton.find_level(x) == l]

        for q in Q:
            neighbours = morton.find_neighbours(q)

            # Add neighbours, and neighbour siblings - may overlap
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
    {np.int64}
        Balanced octree set
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


@numba.njit(
    [types.Long(types.Keys)],
    cache=True
)
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