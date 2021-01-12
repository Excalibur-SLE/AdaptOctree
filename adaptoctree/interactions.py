"""
Calculate the interaction lists for a given tree.

- What is the largest number of nodes in each interaction list possible in the
adaptive tree?
"""
import numba
import numpy as np

import adaptoctree.morton as morton


def find_u(key, leaves):
    """
    Defined for all leaf octants, defined as all adjacent leaf octants including
        the node itself.
    """
    u = []

    # 1. find all neighbours of leaf at same level in the tree
    neighbours = morton.find_neighbours(key)

    # 2. find all adjacent neighbours of key at higher level
    parent_neighbours = morton.find_parent(neighbours)

    # 3. find all adjacent neighbours of key at lower level
    neighbour_children = None
    for neighbour in neighbours:
        if neighbour_children is None:
            neighbour_children = morton.find_children(neighbour)
        else:
            neighbour_children = np.hstack(
                (neighbour_children, morton.find_children(neighbour))
                )

    all_neighbours = np.hstack((neighbours, parent_neighbours, neighbour_children))

    for neighbour in all_neighbours:
        if neighbour in leaves and morton.are_adjacent(neighbour, key):
            u.append(neighbour)

    return np.array(u, dtype=np.int64)


def find_v(key, complete_tree):
    """
    Colleagues are defined as as adjacent nodes at the same level. Defined by
        children of colleagues which are not adjacent to the current node.
        The V list consist of the children of the colleague's of B's parent which
        are not adjacent to B
    """
    v = []
    parent = morton.find_parent(key)

    parent_neighbours = morton.find_neighbours(parent)

    for neighbour in parent_neighbours:
        if neighbour in complete_tree:
            neighbour_children = morton.find_children(neighbour)
            for child in neighbour_children:
                if child in complete_tree and not morton.are_adjacent(child, key):
                    v.append(child)

    return np.array(v, dtype=np.int64)


def find_w(key, leaves):
    """
    Defined for all leaf octants, as all octants which  are descendents of
        colleagues of the node, are not adjacent to the node themselves, but
        their parents are adjacent to the node.
    """

    # 1. Find colleagues
    colleagues = morton.find_neighbours(key)
    w = []

    # Due to balance constraint, only children of colleagues could possibly
    # be in the w list
    for colleague in colleagues:
        children = morton.find_children(colleague)
        for child in children:
            if child in leaves and not morton.are_adjacent(key, child):
                w.append(child)

    return np.array(w, dtype=np.int64)


def find_x(key, leaves, tree):
    """
    Defined for all leaves, for B consists of all octants A which have B on their
    W list.
    """

    x = []

    for leaf in leaves:
        if key in tree[leaf].w:
            x.append(key)

    return np.array(x, dtype=np.int64)
