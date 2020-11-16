"""
Construct an adaptive linear octree form a set of points.

Problems in implementation:
---------------------------

1) Lists used in filtering for balancing algortihm for nodes at a give level.
Can be fixed using another data structure holding an index pointer for a given
level.
2) Sibling checker involves a specific loop.
3) Don't associate points with new nodes, this means that we can't construct
another Octree class for the balanced tree. This can be done at the end, after
the balanced nodes have been found - they can be 'filled'.

"""
import numba
import numpy as np

import adaptoctree.morton as morton


def balance(octree):
    """
    Single-node sequential tree balancing. Based on Algorithm 8 in Sundar et al
        (2012).

    Parameters:
    -----------
    octree : Octree

    Returns:
    --------
    Octree
    """

    depth = octree.depth

    W = octree.tree

    P = None
    balanced = None

    for level in range(depth, 0, -1):

        # Working list, filtered to current level
        Q = W[W[:,1] == level]

        # Q.sort()
        T = []
        len_Q, _ = Q.shape
        T_mask = np.zeros(len_Q, dtype=bool)

        parents = set()

        for i, q in enumerate(Q):
            parent = morton.find_parent(q[0])
            if parent not in parents:
                T_mask[i] = True
                parents.add(parent)

        T = Q[T_mask]

        for t in T:
            siblings = morton.find_siblings(t[0])
            neighbours = morton.find_neighbours(morton.find_parent(t[0]))

            sibling_levels = morton.find_level(siblings)
            neighbour_levels = morton.find_level(neighbours)

            tmp_siblings = np.c_[siblings, sibling_levels]
            tmp_neigbours = np.c_[neighbours, neighbour_levels]

            if balanced is None:
                balanced = tmp_siblings
            else:
                balanced = np.r_[balanced, tmp_siblings]

            if P is None:
                P = tmp_neigbours
            else:
                P = np.r_[P, tmp_neigbours]

        # Remove duplicates in P, if they exist
        P = np.r_[P, W[W[:,1]==(level-1)]]
        if P.shape[0] > 0:
            P = np.unique(P, axis=0)

        W = np.r_[W, P]
        P = None

    # Sort and linearise
    tmp = np.sort(balanced[:,0])
    return linearise(tmp)


def linearise(tree):
    """
    Remove overlaps in a sorted linear tree. Algorithm 7 in Sundar (2012).

    Parameters:
    -----------
    tree : np.array(dtype=np.int64)

    Returns:
    --------
    np.array(np.int64)
    """

    mask = np.zeros_like(tree, dtype=bool)

    n_octants = tree.shape[0]

    for i in range(n_octants-1):
        if morton.not_ancestor(tree[i], tree[i-1]):
            mask[i] = True

    return tree[mask]


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
    sources : np.array(shape=(nsources, 3), dtype=np.float32)
    targets : np.array(shape=(nsources, 3), dtype=np.float32)
    maximum_level : np.int32
        Maximum level of the octree.
    maximum_particles : np.int32
        Maximum number of particles per node.

    Returns:
    --------
    Octree
        Unbalanced adaptive linear Octree.
    """

    max_bound, min_bound = morton.find_bounds(sources, targets)
    octree_center = morton.find_center(max_bound, min_bound)
    octree_radius = morton.find_radius(octree_center, max_bound, min_bound)

    tree = []

    built = False
    level = 1
    size = 1

    leaf_index = 0

    while not built:

        if (level == (maximum_level)):
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
                tree.append((leaf, level))
                size += 1

        if (not refined_sources) or (not refined_targets):
            depth = level
            built = True

        else:
            sources = np.concatenate(refined_sources)
            targets = np.concatenate(refined_targets)

        level += 1

    return np.array(tree), depth, size
