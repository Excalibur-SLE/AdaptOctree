"""
Construct an adaptive linear octree form a set of points.
"""
import numba
import numpy as np

import adaptoctree.morton as morton


def remove_duplicates(a):
    """
    Dynamically de-dupe sorted list
    """

    res = []

    tmp = None

    for i, v in enumerate(a):
        if v != tmp:
            res.append(v)
        tmp = a[i]

    return res


def balance(octree):

    depth = octree.depth

    tree = [n.key for n in octree.tree]

    balanced = []

    for level in range(depth, 0, -1):

        # Working list
        W = []

        # Create working list of leaves at current level
        for key in tree:

            if morton.find_level(key) == level:
                W.append(key)

        # For each node in working list




# def balance(octree, maximum_depth):
#     """
#     Balance a sorted linear octree sequentially. Algorithm 8 in Sundar (2012).

#     Parameters:
#     -----------
#     octree : Octree

#     Returns:
#     --------
#     None
#     """

#     # Start algorithm at root

#     # Working list
#     W = octree.tree

#     # Need to pick up indices corresponding to level data in the working list

#     # Temporary Buffer
#     P = []

#     # Final balanced tree
#     R = []

#     for level in range(maximum_depth, 1, -1):
#         # Get subset of working list at this level
#         Q = []

#         # Iterates over whole linear octree ....
#         for w in W:
#             if morton.find_level(w) == level:
#                 Q.append(w)

#         # Sort(Q), but should be sorted
#         # Exclude siblings to reduce extra work
#         T = []
#         for q in Q:
#             if not T:
#                 T.append(q)
#             else:
#                 for t in T:
#                     if not morton.not_sibling(a, t):
#                         T.append(q)

#         for t in T:
#             R = R + morton.find_siblings(t)
#             P = P + morton.find_neighbours(morton.find_parent(t))

#         # Update Buffer and working list
#         W_new = []
#         for w in W:
#             if morton.find_level(w) == (level - 1):
#                 P.append(w)
#             else:
#                 W_new.append(w)

#         # Update working list
#         W = W_new
#         W = W + remove_duplicates(P)

#         # Reset buffer
#         P = []

#     # Sort R

#     # Linearise
#     linearise(R)

#     return R

def linearise(octree):
    """
    Remove overlaps in a sorted tree. Algorithm 7 in Sundar (2012).

    Parameters:
    -----------
    octree : Octree

    Returns:
    --------
    None
    """
    linearised = []
    size = 1

    for i in range(len(octree)-1):
        if morton.not_ancestor(octree[i].key, octree[i+1].key):
            linearised.append(octree[i])
            size += 1

    linearised.append(octree[-1])

    octree.tree = linearised
    octree.size = size


class Node:
    """
    Minimal octree node.
    """
    def __init__(self, key, sources, targets, children=None):
        self.key = key
        self.sources = sources
        self.targets = targets
        if children is not None:
            self.children = children

    def __repr__(self):
        return f"<morton_id>{self.key}</morton_id>"


class Octree:
    """Minimal, list-like, octree"""

    def __init__(self, sources, targets, maximum_level, maximum_particles):

        self.tree, self.depth, self.size = build_tree(
            sources=sources,
            targets=targets,
            maximum_level=maximum_level,
            maximum_particles=maximum_particles
            )

        self.maximum_level = maximum_level
        self.sources = sources
        self.targets = targets

    def __repr__(self):
        return f"<tree>" \
               f"<maximum_level>{self.maximum_level}</maximum_level>" \
               f"<depth>{self.depth}</depth>"\
               f"</tree>"

    def __getitem__(self, key):
        return self.tree[key]

    def __setitem__(self, key, value):
        self.tree[key] = value

    def __len__(self):
        return len(self.tree)


def build_tree(
    sources,
    targets,
    maximum_level,
    maximum_particles,
    ):
    """
    Top-down construction of an adaptive octree mesh.

    Parameters:
    -----------
    sources : np.array(shape=(nsources, 3))
    targets : np.array(shape=(nsources, 3))
    maximum_level : np.int32
        Maximum level of the octree.
    maximum_particles : np.int32
        Maximum number of particles per node.

    Returns:
    --------
    [Node]
        Adaptive linear Octree.
    """

    max_bound, min_bound = morton.find_bounds(sources, targets)
    octree_center = morton.find_center(max_bound, min_bound)
    octree_radius = morton.find_radius(octree_center, max_bound, min_bound)

    tree = []

    built = False
    level = 1
    size = 1

    leaf_index = 0
    level_index_ptr = []

    while not built:

        if (level == (maximum_level + 1)):
            depth = maximum_level
            built = True

        # Heavy lifting
        source_keys = morton.encode_points(sources, level, octree_center, octree_radius)
        target_keys = morton.encode_points(targets, level, octree_center, octree_radius)

        particle_keys = np.hstack((source_keys, target_keys))
        particle_index_array = np.argsort(particle_keys)
        unique_keys, counts = np.unique(particle_keys, return_counts=True) # O(N)

        refined_sources = []
        refined_targets = []

        for i, count in enumerate(counts):
            leaf = unique_keys[i]

            source_idxs = np.where(source_keys == leaf)
            target_idxs = np.where(target_keys == leaf)

            if count > maximum_particles:
                refined_sources.append(sources[source_idxs])
                refined_targets.append(targets[target_idxs])

            else:
                # Need to keep a track of for the level index pointer
                leaf_index += 1

                tree.append(
                    Node(
                        key=leaf,
                        sources=sources[source_idxs],
                        targets=targets[target_idxs]
                        )
                    )

                size += 1

        if (not refined_sources) or (not refined_targets):
            depth = level
            built = True

        else:
            sources = np.concatenate(refined_sources)
            targets = np.concatenate(refined_targets)

        level_index_ptr.append(leaf_index)
        level += 1

    return tree, depth, size
