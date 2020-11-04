"""
Balancing algorithms for adaptive octrees.
"""
import time

import numpy as np

from adaptoctree.tree import Octree
import adaptoctree.morton as morton


def balance(octree):
    """
    Conduct 2:1 balance refinement on an unbalanced linear octree pre-sorted by
    Morton key.

    Parameters:
    -----------
    octree : Octree

    Returns:
    --------
    Octree
    """

    refined = []

    # Examine tree nodes iterating up through levels, ensuring that balance
    # condition always observed

    balanced = False

    idx = octree.size-1

    print(f"size {octree.size}")

    while not balanced:
        # Examine nodes beginning with smallest by Morton Key
        current = octree[idx]

        # Generate neighbours of this octant
        neighbours = morton.find_neighbours(current)

        idx = max(0, idx-1)
        if current.key == 0:
            balanced = True

    level = morton.find_level(octree[-1].key)
    # print(f"depth {level}")

    return refined


