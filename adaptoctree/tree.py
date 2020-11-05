"""
Construct an adaptive linear octree form a set of points.
"""
import numba
import numpy as np

import adaptoctree.morton as morton




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
        if not morton.is_ancestor(octree[i].key, octree[i+1].key):
            linearised.append(tree[i])

    octree.tree = linearised


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

        self.tree, self.depth, self.size, self.working_set = build_tree(
            sources=sources,
            targets=targets,
            maximum_level=maximum_level,
            maximum_particles=maximum_particles
            )

        self.maximum_level = maximum_level

    def __repr__(self):
        return f"<tree>" \
               f"<maximum_level>{self.maximum_level}</maximum_level>" \
               f"<depth>{self.depth}</depth>"\
               f"</tree>"

    def __getitem__(self, key):
        return self.tree[key]

    def __setitem__(self, key, value):
        self.tree[key] = value


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

    tree = [Node(0, sources, targets)]
    working_set = set([0])

    built = False
    level = -1

    size = 1

    while not built:
        level += 1

        if (level == maximum_level):
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
            if count > maximum_particles:
                source_idxs = np.where(source_keys == leaf)
                target_idxs = np.where(target_keys == leaf)
                refined_sources.append(sources[source_idxs])
                refined_targets.append(targets[target_idxs])

            else:
                source_idxs = np.where(source_keys == leaf)
                target_idxs = np.where(target_keys == leaf)

                tree.append(
                    Node(
                        key=leaf,
                        sources=sources[source_idxs],
                        targets=targets[target_idxs]
                        )
                    )

                working_set.add(leaf)
                size += 1

        if (not refined_sources) or (not refined_targets):
            built = True

        else:
            sources = np.concatenate(refined_sources)
            targets = np.concatenate(refined_targets)

    return tree, level, size, working_set
