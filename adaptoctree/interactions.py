"""
Calculate the interaction lists for a given tree.

- What is the largest number of nodes in each interaction list possible in the
adaptive tree?
"""
import numba
import numpy as np

import adaptoctree.morton as morton


def find_u(key, tree):
    """
    Defined for all leaf octants, defined as all adjacent leaf octants including
        the node itself.
    """
    u = np.zeros(shape=(27,), dtype=np.int64)

    # 1. find all neighbours of leaf at same level in the tree
    neighbours = morton.find_neighbours(key)

    # 2. find all neighbours of key at higher level
    parent_neighbours = morton.find_parent(neighbours)

    all_neighbours = np.hstack((neighbours, parent_neighbours))

    i = 0
    for neighbour in all_neighbours:
        if neighbour in tree:
            u[i] = neighbour
            i += 1

    return u[:i]


def find_v(key, tree):
    """
    Colleagues are defined as as adjacent nodes at the same level. Defined by
        children of colleagues which are not adjacent to the current node
    """

    #1. Find colleagues
    neighbours = morton.find_neighbours(key)
    colleagues = -1*np.ones_like(neighbours)
    v = -1*np.ones(shape=(32,), dtype=np.int64)

    n_colleagues = 0
    for i, neighbour in enumerate(neighbours):
        if neighbour in tree:
            colleagues[i] = neighbour
            n_colleagues += 1

    if n_colleagues == 0:
        return v

    idx = 0
    for colleague in colleagues:
        children = morton.find_children(colleague)
        for child in children:
            if child in tree and not morton.are_adjacent(child, key):
                v[idx] = child
                idx += 1

    return v


def find_w(key, tree):
    """
    Defined for all leaf octants, as all octants which  are descendents of
        colleagues of the node, are not adjacent to the node themselves, but
        their parents are adjacent to the node.
    """

    #1. Find colleagues
    neighbours = morton.find_neighbours(key)
    colleagues = -1*np.ones_like(neighbours)
    w = -1*np.ones(shape=(32,), dtype=np.int64)

    n_colleagues = 0
    for i, neighbour in enumerate(neighbours):
        if neighbour in tree:
            colleagues[i] = neighbour
            n_colleagues += 1



def find_x(key, tree):
    pass
