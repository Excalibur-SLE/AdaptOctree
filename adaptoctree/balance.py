"""
Balancing algorithms for adaptive octrees.
"""
import time

import numpy as np

from adaptoctree.morton import find_level
from adaptoctree.tree import Octree


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
        print(current)
        idx = max(0, idx-1)
        if current.key == 0:
            balanced = True

    level = find_level(octree[-1].key)
    # print(f"depth {level}")

    return refined


if __name__ == "__main__":
    np.random.seed(0)

    N = int(50)
    # sources = targets = make_moon(N)
    sources = targets = np.random.rand(N, 3)

    tree_conf = {
        "sources": sources,
        "targets": targets,
        "maximum_level": 1,
        "maximum_particles": 5
    }

    # Sort sources and targets by octant at level 1 of octree
    start = time.time()
    tree = Octree(**tree_conf)

    print(tree)

    # balance(tree)
