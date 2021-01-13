"""
Construct an adaptive balanced linear octree form a set of points.
"""

import numba
import numpy as np

import adaptoctree.morton as morton
import adaptoctree.types as types
import adaptoctree.interactions as interactions


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
    {np.int64}
        Balanced octree
    """
    return remove_overlaps(balance_subroutine(tree, depth), depth)


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
    cache=True
)
def complete_tree(balanced):
    """
    Take a balanced tree, and complete it - adding all of its ancestors.
    """
    complete = set(balanced)

    for key in balanced:
        complete.update(morton.find_ancestors(key))

    return complete


def populate_leaves(points, points_to_leaves, complete_tree_set,
                    leaves_set, leaves_arr, depth, x0, r0):
    """
    Populate a data structure encapsulating each leaf key, the coordinates of
        the points contained within them, and their interaction lists: U, V, W
        and X. The X list is an inversion of the mapping contained in the W
        list, and is therefore computed on the fly from the results of the W
        list computation.

    Parameters:
    -----------
    points : np.array(shape=(N, 3))
        Coordinates of all N points.
    points_to_leaves : np.array(shape=(N,))
        Morton key corresponding to each point.
    complete_tree_set : {np.int64}
        Morton keys of all nodes in tree.
    leaves_set : {np.int64}
        Morton keys of all leaf nodes.
    leaves_arr : np.array(dtype=np.int64)
        Morton keys of all leaf nodes.
    depth : np.int64
        Depth of the valanced tree.
    x0 : np.array(shape=(3,), dtype=np.float64)
        Center of root node of Octree.
    r0 : np.float64
        Half side length of root node

    Returns:
    --------
    dict(np.int64, Node)
        Dictionary of leaf nodes, keys corresponding to leaf Morton keys, and
        values to Node objects.
    """
    pop = dict()

    # Temporary structure to store X lists
    x_tmp = {leaf:set() for leaf in leaves_arr}

    for leaf in leaves_arr:
        points_subset = points[points_to_leaves == leaf]

        u = interactions.find_u(leaf, leaves_set)
        v = interactions.find_v(leaf, complete_tree_set)
        w = interactions.find_w(leaf, leaves_set)
        pop[leaf] = Node(key=leaf, points=points_subset, u=u, v=v, w=w, x=None)

        # Update X list
        for key in w:
            if x_tmp[key] is not None:
                x_tmp[key].update(w)
            else:
                x_tmp[key].add(w)

    # Store X list
    for leaf, x_list in x_tmp.items():
        pop[leaf].x = x_list

    return pop


def populate_tree(populated_leaves, complete_tree_set, depth):
    """
    Populate whole tree with interaction lists, from the leaves.
    """
    pop = {level: dict() for level in range(0, depth+1)}

    for leaf, node in populated_leaves.items():
        level = morton.find_level(leaf)
        pop[level][leaf] = node
        parent = morton.find_parent(leaf)
        parent_level = morton.find_level(parent)
        while parent > -1:
            v = interactions.find_v(parent, complete_tree_set)
            leaf_points = node.points

            if parent not in pop[parent_level]:
                pop[parent_level][parent] = Node(
                    u=None, w=None, v=v, x=None, key=parent, points=leaf_points)
            else:
                parent_points = pop[parent_level][parent].points
                points = np.vstack((leaf_points, parent_points))
                pop[parent_level][parent].points = points

            parent = morton.find_parent(parent)
            parent_level = morton.find_level(parent)

    return pop


class Node:
    """
    API for tree node.
    """
    def __init__(self, key, points, u, v, w, x):
        self.key = key
        self.points = points
        self.u = u
        self.v = v
        self.w = w
        self.x = x

    def __repr__(self):
        return f"<Node key={self.key}>"


class Tree:
    """
    API for tree, for use with PyExaFMM.
    """
    def __init__(self, points, max_level, max_points, start_level):
        self.particles = points
        _unbalanced = build(points, max_level, max_points, start_level)
        self.depth = find_depth(_unbalanced)

        max_bound, min_bound = morton.find_bounds(points)
        self.center= morton.find_center(max_bound, min_bound)
        self.radius = morton.find_radius(self.center, max_bound, min_bound)

        balanced_set = balance(_unbalanced, self.depth)
        balanced_arr = np.fromiter(balanced_set, np.int64)
        points_to_leaves = points_to_keys(
            points, balanced_arr,
            self.depth,
            self.center,
            self.radius
            )
        non_empty_balanced_arr = np.unique(points_to_leaves)
        non_empty_balanced_set = set(non_empty_balanced_arr)
        complete_tree_set = complete_tree(non_empty_balanced_arr)

        # Leaves populated before this attribute assignment
        populated_leaves = populate_leaves(
            points,
            points_to_leaves,
            complete_tree_set,
            non_empty_balanced_set,
            non_empty_balanced_arr,
            self.depth,
            self.center,
            self.radius
            )

        self.tree = populate_tree(
            populated_leaves,
            complete_tree_set,
            self.depth
            )

    def __getitem__(self, level):
        """Read only"""
        return self.tree[level]
