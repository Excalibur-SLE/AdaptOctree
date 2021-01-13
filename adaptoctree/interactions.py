"""
Calculate the interaction lists for a given tree.
"""
import numba
import numpy as np

import adaptoctree.morton as morton
import adaptoctree.types as types


@numba.njit(
    [types.KeySet(types.Key, types.Keys, types.Coord, types.Double)],
    cache=True
)
def find_u(key, leaves, x0, r0):
    """
    U List. Defined for all leaves. For a leaf B, it consists of all leaves
        adjacent (share a vertex, face or side) to B, including B itself. In the
        FMM algorithm these represent the near interactions of the target leaf
        B, and are calculated through direct summation with the source points
        and the kernel.

        Strategy: The 2:1 balance constraint ensures that neighbours are within
        1 level of the target key. Therefore, we simply compute valid neighbours
        , which is rapid as they are defined algebraicly, and check whether they
        are leaves, by set inclusion.
    """

    leaves_set = set(leaves)
    u = set()

    # 1. Find all neighbours of leaf at same level in the tree
    neighbours = morton.find_neighbours(key)

    # 2. Find all adjacent neighbours of key at higher level
    parent_neighbours = morton.find_parent(neighbours)

    # 3. Find all adjacent neighbours of key at lower level
    neighbour_children = np.array([-1], dtype=np.int64)
    for neighbour in neighbours:
        neighbour_children = np.hstack((neighbour_children, morton.find_children(neighbour)))

    neighbour_children = neighbour_children[1:]

    all_neighbours = np.hstack((neighbours, parent_neighbours))

    for neighbour in all_neighbours:
        if neighbour in leaves_set and morton.are_adjacent(neighbour, key, x0, r0):
                u.add(neighbour)

    return u


@numba.njit(
    [types.KeySet(types.Key, types.Keys, types.Coord, types.Double)],
    cache=True
)
def find_v(key, complete_tree, x0, r0):
    """
    V List. Defined for all nodes in the tree. Colleagues are defined as
        adjacent octants at the same level. The V list consists of all children
        of the colleagues of the target octant B which are not adjacent to B.
    """
    v = set()
    complete_tree_set = set(complete_tree)
    parent = morton.find_parent(key)

    parent_neighbours = morton.find_neighbours(parent)

    for neighbour in parent_neighbours:
        neighbour_children = morton.find_children(neighbour)
        for child in neighbour_children:
            if child in complete_tree_set and not morton.are_adjacent(child, key, x0, r0):
                v.add(child)

    return v


@numba.njit(
    [types.KeySet(types.Key, types.Keys, types.Coord, types.Double)],
    cache=True
)
def find_w(key, leaves, x0, r0):
    """
    W List. Defined for all leaves. For a leaf B, the leaf A is in its W list
        iff A is a descendent of a colleague of B, A is not adjacent to B,
        and the parent of A is adjacent to B.

        Strategy: Due to the 2:1 balance constraint, only the children of
        colleagues could possibly be in the W list, otherwise the constraint
        is violated. This means we simply have to search through the children
        of colleagues of a target B, and identify the ones which are not
        adjacent to it.
    """

    # 1. Find colleagues
    colleagues = morton.find_neighbours(key)
    w = set()
    leaves_set = set(leaves)

    # 2. Find non-adjacent colleague-children
    for colleague in colleagues:
        children = morton.find_children(colleague)
        for child in children:
            if child in leaves_set and not morton.are_adjacent(key, child, x0, r0):
                w.add(child)

    return w
