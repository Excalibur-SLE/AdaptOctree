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
    level_index_pointer = octree.level_index_pointer

    P = []
    balanced = []

    for level in range(depth, 0, -1):

        # Working list
        Q = []

        # Create working list of leaves at current level
        # Need efficient level filter
        for w in W:
            if morton.find_level(w) == level:
                Q.append(w)

        Q.sort()

        T = []
        for q in Q:
            siblings = morton.find_siblings(q)
            siblings_in_T = False

            for sibling in siblings:
                if sibling in T:
                    siblings_in_T = True

            if not siblings_in_T:
                T.append(q)

        for t in T:
            balanced = balanced + list(morton.find_siblings(t))
            P = P + list(morton.find_neighbours(morton.find_parent(t)))

        # Need efficient level filter
        for w in W:
            if morton.find_level(w) == (level-1):
                P.append(w)

        # Remove duplicates in P
        P.sort()
        P = remove_duplicates(P)

        W = W + P
        P = []

    balanced.sort()
    balanced = linearise(balanced)

    return balanced


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

    for i in range(len(octree)-1):
        if morton.not_ancestor(octree[i], octree[i+1]):
            linearised.append(octree[i])

    linearised.append(octree[-1])

    return linearised


class Octree:
    """Minimal, list-like, octree"""

    def __init__(self, sources, targets, maximum_level, maximum_particles):

        self.tree, self.depth, self.size, self.level_index_pointer = build_tree(
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

    NOTE: level_index_pointer starts from level 1 NOT level 0, as this is where
        Octree construction starts.

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
    level_index_pointer = [leaf_index]

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
                tree.append(leaf)
                size += 1

        if (not refined_sources) or (not refined_targets):
            depth = level
            built = True

        else:
            sources = np.concatenate(refined_sources)
            targets = np.concatenate(refined_targets)

        level_index_pointer.append(leaf_index)
        level += 1

    return tree, depth, size, level_index_pointer
