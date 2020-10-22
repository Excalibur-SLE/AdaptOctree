"""
Construct z order locational code to describe a set of points.
"""
import numba
import numpy as np



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


@numba.njit
def interleave(x, y, z):
    """
    Interleave three 32-bit integers, bitwise.

    Parameters:
    -----------
    x : np.int32
    y : np.int32
    z : np.int32

    Returns:
    --------
    interleaved : np.int64
    """

    interleaved = np.int64(0)

    for i in range(32):
        xi = x & (1 << i)
        yi = y & (1 << i)
        zi = z & (1 << i)

        interleaved |= xi << (i+2)
        interleaved |= yi << (i+1)
        interleaved |= zi << i

    return interleaved


@numba.njit
def bit_length(x):

    res = 0

    while x > 0:
        x = x >> 1
        res += 1

    return res


@numba.njit
def morton_encode(anchor):
    """
    Use a node's anchor to find it's Morton encoding.

    Parameters:
    -----------
    anchor : np.array(shape=(4,), dtype=np.int32)

    Returns:
    --------
    np.int64
    """

    # Interleave anchor bits
    interleaved = interleave(anchor[0], anchor[1], anchor[2])

    # Append level
    interleaved = interleaved << bit_length(anchor[-1]) | anchor[-1]

    return interleaved
