"""
Construct an adaptive linear octree form a set of points.
"""
import numba
import numpy as np


def compute_bounds(sources, targets):
    """
    Compute bounds of computational domain of an Octree containing given
        sources and targets. This method iterates through all sources and targets
        to find max/min so will need to be parallelised, complexity O(N).

    Parameters:
    -----------
    sources : np.array(shape=(nsources, 3), dtype=np.float64)
    targets : np.array(shape=(ntargets, 3), dtype=np.float64)
    Returns:
    --------
    (np.array(shape=(3,), dtype=np.float64), np.array(shape=(3,), dtype=np.float64))
        Tuple containing the maximal/minimal coordinate in the sources/targets
        provided.
    """

    min_bound = np.min(
        np.vstack([np.min(sources, axis=0), np.min(targets, axis=0)]), axis=0
    )

    max_bound = np.max(
        np.vstack([np.max(sources, axis=0), np.max(targets, axis=0)]), axis=0
    )

    return max_bound, min_bound


def compute_center(max_bound, min_bound):
    """
    Compute center of Octree's root node.

    Parameters:
    -----------
    max_bound : np.array(shape=(3,) dtype=np.float64)
        Maximal point in Octree's root node.
    min_bound : np.array(shape=(3,) dtype=np.float64)
        Minimal point in Octree's root node.
    Returns:
    --------
    np.array(shape=(3,), dtype=np.float64)
        Cartesian coordinates of center.
    """

    center = (min_bound + max_bound) / 2

    return center


def compute_radius(center, max_bound, min_bound):
    """
    Compute half side length of Octree's root node.
    Parameters:
    ----------
    center : np.array(shape=(3,) dtype=np.float64)
    max_bound : np.array(shape=(3,) dtype=np.float64)
        Maximal point in Octree's root node.
    min_bound : np.array(shape=(3,) dtype=np.float64)
        Minimal point in Octree's root node.:
    Returns:
    --------
    np.float64
    """

    factor = 1 + 1e-5
    radius = np.max([np.max(center - min_bound), np.max(max_bound - center)]) * factor

    return radius


@numba.njit(cache=True)
def get_4d_index_from_point(point, level, x0, r0):
    """
    Get 4D index from point in 3 dimensions contained in the computational
        domain defined by an Octree with a root node center at x0 and a root
        node radius of r0. This method is only valid for points known to be in
        the Octree's computational domain.
    Parameters:
    -----------
    point : np.array(shape=(3,), dtype=np.float64)
    level : np.int64
    x0 : np.array(shape=(3,))
        The center of the Octree's root node.
    r0: np.float64
        The half side length of the Octree's root node
    Returns:
    --------
    np.array(shape=(4,), dtype=np.int64)
    """
    index = np.empty(4, dtype=np.int64)
    index[3] = level
    xmin = x0 - r0

    side_length = 2 * r0 / (1 << level)
    index[:3] = np.floor((point - xmin) / side_length).astype(np.int64)

    return index


@numba.njit(cache=True)
def get_key_from_4d_index(index):
    """
    Compute Morton key from a 4D index, The 4D index is composed as
        [xidx, yidx, zidx, level], corresponding to the physical index of a node
        in a partitioned box. This method works by calculating the octant
        coordinates at level 1 [x, y, z] , where x,y,z ∈ {0, 1}, and appending
        the resulting bit value `xyz` to the key. It continues to do this until
        it reaches the maximum level of the octree. Finally it adds a level
        offset to ensure that the keys at each level are unique.
    Parameters:
    -----------
    index : np.array(shape=(4,), type=np.int64)
    Returns:
    --------
    int
    """
    max_level = index[-1]
    key = 0
    for level in range(max_level):
        key |= (index[2] & (1 << level)) << 2 * level
        key |= (index[1] & (1 << level)) << 2 * level + 1
        key |= (index[0] & (1 << level)) << 2 * level + 2

    key += get_level_offset(max_level)

    return key


def get_key_from_point(point, level, x0, r0):
    """
    Get Morton key from Cartesian coordinates of a point in the computational
        domain of a given Octree.

    Parameters:
    -----------
    point : np.array(shape=(3,), dtype=np.float64)
    level : np.int64
        The level at which the key is being calculated
    x0 : np.array(shape=(3,))
        The center of the Octree's root node.
    r0: np.float64
        The half side length of the Octree's root node
    Returns:
    --------
    np.int64
    """
    vec = get_4d_index_from_point(point, level, x0, r0)
    return get_key_from_4d_index(vec)


@numba.njit(cache=True)
def get_level_offset(level):
    """
    The `offset` of a level is determined as the starting starting point of the
    Morton keys for a given level, so that they don't collide with keys from
    previous levels.
    Parameters:
    -----------
    level : int
    Returns:
    --------
    int
    """

    return ((1 << 3 * level) - 1) // 7


class Node:
    """
    Simple tree node.
    """
    def __init__(self, key, sources, targets):
        self.key = key
        self.sources = sources
        self.targets = targets




@numba.njit(cache=True)
def get_keys_from_points(points, level, x0, r0):
    """
    Get Morton keys from array of points in computational domain of a given
        Octree.
    Parameters:
    -----------
    points : np.array(shape=(3,n), dtype=np.float64)
        An array of `n` points.
    level : np.int64
        The level at which the key is being calculated
    x0 : np.array(shape=(3,))
        The center of the Octree's root node.
    r0: np.float64
        The half side length of the Octree's root node
    Returns:
    --------
    np.array(n, dtype=np.int64)
    """
    npoints = len(points)
    keys = np.empty(npoints, dtype=np.int64)
    indices = np.empty((npoints, 4), dtype=np.int64)
    indices[:, -1] = level
    xmin = x0 - r0
    side_length = 2 * r0 / (1 << level)
    indices[:, :3] = np.floor((points - xmin) / side_length).astype(np.int64)
    for i in range(npoints):
        keys[i] = get_key_from_4d_index(indices[i, :])
    return keys


class Node:
    def __init__(self, key, sources, targets, children=None):
        self.key = key
        self.sources = sources
        self.targets = targets
        if children is not None:
            self.children = children

    def __repr__(self):
        return f"ID: {self.key}"



def compute_nodes(sources, targets, level, x0, r0):
    """
    Compute non-empty nodes at a given level, and return counter.

    Parameters:
    -----------
    sources : np.array(shape=(nsources, 3))
        Sorted by morton id at previous level
    targets : np.array(shape=(ntargets, 3))
        sorted by morton id at previous level
    """

    source_keys = get_keys_from_points(sources, level, x0, r0) # O(N)
    target_keys = get_keys_from_points(targets, level, x0, r0) # O(N)

    values, counts = np.unique(source_keys, return_counts=True)


def build_tree(sources, targets, maximum_level, maximum_particles, x0, r0):

    tree = [Node(0, sources, targets)]
    built = False
    level = 0

    while not built:

        if (level == maximum_level):
            built = True

        source_keys = get_keys_from_points(sources, level, x0, r0)
        target_keys = get_keys_from_points(targets, level, x0, r0)

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
                        key=unique_keys[i],
                        sources=sources[source_idxs],
                        targets=targets[target_idxs]
                        )
                    )

        level += 1

        if (not refined_sources) or (not refined_targets):
            built = True

        else:
            sources = np.concatenate(refined_sources)
            targets = np.concatenate(refined_targets)

    return tree



@numba.njit(cache=True)
def get_4d_index_from_key(key):
    """
    Compute the 4D index from a Hilbert key. The 4D index is composed as
        [xidx, yidx, zidx, level], corresponding to the physical index of a node
        in a partitioned box.
    Parameters:
    -----------
    key : int
        Hilbert key
    Returns:
    --------
    index : np.array(shape=(4,), type=np.int64)
    """
    max_level = get_level(key)
    key = key - get_level_offset(max_level)
    index = np.zeros(4, np.int64)
    index[3] = max_level
    for level in range(max_level):
        index[2] |= (key & (1 << 3 * level)) >> 2 * level
        index[1] |= (key & (1 << 3 * level + 1)) >> (2 * level + 1)
        index[0] |= (key & (1 << 3 * level + 2)) >> (2 * level + 2)
    return index


@numba.njit(cache=True)
def get_center_from_4d_index(index, x0, r0):
    """
    Get center of given Octree node described by a 4d index.
    Parameters:
    -----------
    index : np.array(shape=(4,), dtype=np.int64)
        4D index.
    x0 : np.array(shape=(3,))
        Center of root node of Octree.
    r0 : np.float64
        Half width length of root node.
    Returns:
    --------
    np.array(shape=(3,))
    """
    xmin = x0 - r0
    level = index[-1]
    side_length = 2 * r0 / (1 << level)
    return (index[:3] + .5) * side_length + xmin


@numba.njit(cache=True)
def get_center_from_key(key, x0, r0):
    """
    Get (Cartesian) center of node from its Morton ID.
    Parameters:
    -----------
    key : np.int64
    x0 : np.array(shape=(3,))
    r0 : np.float64
    Returns:
    --------
    np.array(shape=(3,))
    """
    index = get_4d_index_from_key(key)
    return get_center_from_4d_index(index, x0, r0)

@numba.njit(cache=True)
def get_level(key):
    """
    Get octree level from Morton ID.
    Parameters:
    -----------
    key : int
    Returns:
    --------
    int
    """
    level = -1
    offset = 0
    while key >= offset:
        level += 1
        offset += 1 << 3 * level
    return level


def compute_vertices(node, x0, r0):
    """
    vertices of node
    """

    center = get_center_from_key(node.key, x0, r0)
    level = get_level(node.key)
    radius = 2 * r0 / (1 << level)

    vertices = np.empty(shape=(8, 3))
    idx = 0
    for i in [-1, 1]:
        for j in [-1, 1]:
            for k in [-1, 1]:
                vertices[idx, :] = np.array([center[0]+i*radius, center[1]+j*radius, center[2]+k*radius])
                idx += 1

    return vertices


def plot_tree(tree, x0, r0):
    """

    Parameters:
    -----------
    tree : [Node]
    """

    import itertools

    import matplotlib.pyplot as plt
    from mpl_toolkits.mplot3d import Axes3D

    points = []

    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')

    unique = []

    for node in tree:
        level = get_level(node.key)
        radius = r0 / (1 << level)

        center = get_center_from_key(node.key, x0, r0)

        r = [-radius, radius]

        for s, e in itertools.combinations(np.array(list(itertools.product(r, r, r))), 2):
            if np.sum(np.abs(s-e)) == r[1]-r[0]:
                ax.plot3D(*zip(s+center, e+center), color="b")

    # Plot particle data
    sources = tree[0].sources
    ax.scatter(sources[:, 0], sources[:, 1], sources[:, 2], c='g', s=0.8)
    plt.show()


def main():
    n = 150

    np.random.seed(0)
    sources = np.random.rand(n, 3)
    targets = sources
    maximum_level = 5
    maximum_particles_per_node = 1

    max_bound, min_bound = compute_bounds(sources, targets)
    x0 = compute_center(max_bound, min_bound)
    r0 = compute_radius(x0, max_bound, min_bound)

    # Sort sources and targets by octant at level 1 of octree
    tree = build_tree(sources, targets, maximum_level, 100, x0, r0)
    print(tree)

    plot_tree(tree, x0, r0)



if __name__ == "__main__":
    main()
