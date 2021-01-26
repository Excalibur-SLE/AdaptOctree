"""
Construct an adaptive balanced linear octree form a set of points.
"""

import numba
import numpy as np

import adaptoctree.morton as morton
import adaptoctree.types as types


@numba.njit(
    [types.LongIntList(types.LongArray, types.Keys, types.Long)],
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
            types.Coords, types.Long, types.Long, types.Coord,
            types.Double, types.Keys, types.LongArray, types.Int
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
    [types.Keys(types.Coords, types.Long, types.Long, types.Long)],
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
    [types.KeySet(types.KeyList, types.Long)],
    cache=True
)
def remove_overlaps(tree, depth):
    """
    Remove the overlaps in a balanced octree.

        Strategy: Traverse the octree level by level, bottom-up, and check if
        any ancestors lie in the tree. If they do, then remove them.
    """

    result = set(tree)

    for level in range(depth, 0, -1):
        work_items = [x for x in tree if morton.find_level(x) == level]

        for work_item in work_items:
            ancestors = morton.find_ancestors(work_item)
            ancestors.remove(work_item)
            for ancestor in ancestors:
                if ancestor in result:
                    result.remove(ancestor)

    return result


@numba.njit(
    [types.KeyList(types.Keys, types.Long)],
    cache=True
)
def balance_subroutine(tree, depth):
    """
    Perform balancing to enforce the 2:1 constraint between neighbouring
        nodes in linear octree.

        Strategy: Traverse the octree level by level, bottom-up, and check if
        the parent's of a given node's parent lie in the tree, add them and their
        respective siblings. This enforces the 2:1 constraint.
    """
    balanced = set(tree)

    for level in range(depth, 0, -1):
        work_items = [x for x in balanced if morton.find_level(x) == level]

        for work_item in work_items:
            neighbours = morton.find_neighbours(work_item)

            for neighbour in neighbours:
                parent = morton.find_parent(neighbour)
                parent_level = level-1

                if ~(neighbour in balanced) and ~(parent in balanced):

                    balanced.add(parent)

                    if parent_level > 0:
                        siblings = morton.find_siblings(parent)
                        balanced.update(siblings)

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
    tmp = remove_overlaps(balance_subroutine(tree, depth), depth)
    return np.fromiter(tmp, np.int64)


@numba.njit(
    [types.Keys(types.Coords, types.Keys, types.Long, types.Coord, types.Double)],
    cache=True
)
def points_to_keys(points, tree, depth, x0, r0):
    """
    Assign particle positions to Morton keys in a given tree.
    """
    n_points = points.shape[0]
    leaves = -1*np.ones(n_points, dtype=np.int64)
    tree_set = set(tree)

    keys = morton.encode_points_smt(points, depth, x0, r0)

    for i, key in enumerate(keys):
        ancestors = morton.find_ancestors(key)

        ints = ancestors.intersection(tree_set)
        if ints:
            leaves[i] = next(iter(ints))

    return leaves


@numba.njit(
    [types.Long(types.Keys)],
    cache=True
)
def find_depth(tree):
    """
    Return maximum depth of a linear octree.
    """
    levels = morton.find_level(np.unique(tree))
    return np.max(levels)



@numba.njit(
    [types.KeySet(types.Keys)],
    cache=True)
def _complete_tree(leaves):
    """
    Internal jit'd function for completing a tree.
    """
    tree_set = set(leaves)

    for leaf in leaves:
        tree_set.update(morton.find_ancestors(leaf))

    return tree_set


def complete_tree(leaves):
    """
    Complete a tree defined by its leaves.

    Parmeters:
    ----------
    leaves : np.array(dtype=np.int64)

    Returns:
    --------
    np.array(dtype=np.int64)
    """
    tree_set = _complete_tree(leaves)
    return np.fromiter(tree_set, dtype=np.int64)


@numba.njit(cache=True)
def find_interaction_lists(leaves, depth):
    """
    Compute the interaction lists for all leaf keys.

    Parameters:
    -----------
    leaves : np.array(dtype=np.int64)
    depth : np.int64

    Returns:
    --------
    (np.array(dtype=np.int64))
    """

    leaves_set = set(leaves)
    u = -np.ones(shape=(len(leaves), 90), dtype=np.int64)
    x = -np.ones(shape=(len(leaves), 9), dtype=np.int64)
    v = -np.ones(shape=(len(leaves), 189), dtype=np.int64)
    w = -np.ones(shape=(len(leaves), 208), dtype=np.int64)

    def find_interaction_list(i, leaves, leaves_set, depth, u, x, v, w):
        """Find interaction lists for a given node"""

        def compute_parent_level(key, leaves_set, colleagues_parents, parent_colleagues, depth):
            """
            Find interaction lists at parent level of node.
            """
            # U List (P2P)
            cp_in_tree = np.zeros_like(colleagues_parents)
            i = 0
            for cp in colleagues_parents:
                if cp in leaves_set:
                    cp_in_tree[i] = cp
                    i += 1

            cp_in_tree = cp_in_tree[:i]

            adj_idxs = morton.are_adjacent_vec(key, cp_in_tree, depth)
            adjacent = cp_in_tree[adj_idxs == 1]

            # X List (P2L)
            pc_in_tree = np.zeros_like(parent_colleagues)
            i = 0
            for pc in parent_colleagues:
                if pc in leaves_set:
                    pc_in_tree[i] = pc
                    i += 1

            pc_in_tree = pc_in_tree[:i]

            not_adj_idxs = morton.are_adjacent_vec(key, pc_in_tree, depth)
            not_adjacent = pc_in_tree[not_adj_idxs == 0]

            return len(adjacent), adjacent, len(not_adjacent), not_adjacent

        def compute_current_level(key, leaves_set, colleagues, parent_colleagues_children, depth):
            """
            Find interaction lists at level of node.
            """
            # U List (P2P)
            c_in_tree = np.zeros_like(colleagues)
            i = 0
            for c in colleagues:
                if c in leaves_set:
                    c_in_tree[i] = c
                    i += 1

            c_in_tree = c_in_tree[:i]
            adj_idxs = morton.are_adjacent_vec(key, c_in_tree, depth)
            adjacent = c_in_tree[adj_idxs == 1]

            # V List (M2L)
            pcc_in_tree = np.zeros_like(parent_colleagues_children)
            i = 0
            for pcc in parent_colleagues_children:
                if pcc in leaves_set:
                    pcc_in_tree[i] = pcc
                    i += 1

            pcc_in_tree = pcc_in_tree[:i]
            adj_idxs = morton.are_adjacent_vec(key, pcc_in_tree, depth)
            not_adjacent = pcc_in_tree[adj_idxs == 0]

            return len(adjacent), adjacent, len(not_adjacent), not_adjacent

        def compute_child_level(key, leaves_set, colleagues_children, depth):
            """
            Find interaction lists at level of node's children.
            """
            # U List (P2P)
            i = 0
            cc_in_tree = np.zeros_like(colleagues_children)
            for cc in colleagues_children:
                if cc in leaves_set:
                    cc_in_tree[i] = cc
                    i += 1

            cc_in_tree = cc_in_tree[:i]
            adj_idxs = morton.are_adjacent_vec(key, cc_in_tree, depth)
            adjacent = cc_in_tree[adj_idxs == 1]

            # W List (M2P)
            not_adjacent = cc_in_tree[adj_idxs == 0]
            return len(adjacent), adjacent, len(not_adjacent), not_adjacent

        u_ptr = 0
        x_ptr = 0
        v_ptr = 0
        w_ptr = 0
        key = leaves[i]
        parent = morton.find_parent(key)
        colleagues = morton.find_neighbours(key)
        colleagues_children = morton.find_children_vec(colleagues).ravel()
        colleagues_parents = np.unique(morton.find_parent(colleagues))
        parent_colleagues = morton.find_neighbours(parent)
        parent_colleagues_children = morton.find_children_vec(parent_colleagues).ravel()

        pu_ptr, padj, px_ptr, pnadj = compute_parent_level(
            key, leaves_set, colleagues_parents, parent_colleagues, depth
        )

        u[i][u_ptr:pu_ptr] = padj
        u_ptr = pu_ptr
        x[i][x_ptr:px_ptr] = pnadj

        cu_ptr, cadj, pcc_ptr, pcc_nadj = compute_current_level(
            key, leaves_set, colleagues, parent_colleagues_children, depth
        )
        u[i][u_ptr:u_ptr+cu_ptr] = cadj
        u_ptr = pu_ptr+cu_ptr
        v[i][v_ptr:pcc_ptr] = pcc_nadj

        ccu_ptr, ccadj, ccw_ptr, ccnadj = compute_child_level(
            key, leaves_set, colleagues_children, depth
        )
        u[i][u_ptr:u_ptr+ccu_ptr] = ccadj
        w[i][w_ptr:ccw_ptr] = ccnadj

    for i in range(len(leaves)):
        find_interaction_list(i, leaves, leaves_set, depth, u, x, v, w)

    return u, x, v, w
