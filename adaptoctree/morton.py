"""
Utility functions to construct Morton encodings from a set of points distributed
in 3D.
"""
import numba
import numpy as np

from adaptoctree.morton_lookup import X_LOOKUP, Y_LOOKUP, Z_LOOKUP


# Number of bits used for level information
LEVEL_DISPLACEMENT = 15

# Mask for last 15 bits
LEVEL_MASK = 0x7fff

# Mask for a lowest order byte
BYTE_MASK = 0xff


@numba.njit(cache=True)
def find_center_from_anchor(anchor, x0, r0):
    """
    Find center of given Octree node from it's anchor.

    Parameters:
    -----------
    anchor : np.array(shape=(4,), dtype=np.uint16)
    x0 : np.array(shape=(3,), dtype=np.float32)
        Center of root node of Octree.
    r0 : np.float32
        Half side length of root node.

    Returns:
    --------
    np.array(shape=(3,))
    """

    xmin = x0 - r0
    level = anchor[3]
    side_length = 2 * r0 / (1 << level)

    return (anchor[:3] + 0.5) * side_length + xmin


@numba.njit(cache=True)
def find_center_from_key(key, x0, r0):
    """
    Find the center of a given Octree node from it's Morton key.

    Parameters:
    -----------
    key : np.int64
        Morton key.
    x0 : np.array(shape=(3,), dtype=np.float32)
        Center of root node of Octree.
    r0 : np.float32
        Half side length of root node.

    Returns:
    --------
    np.array(shape=(3,))
    """
    anchor = decode_key(key)
    return find_center_from_anchor(anchor, x0, r0)


# @numba.njit(cache=True)
def find_level(key):
    """
    Find the last 15 bits of a key, corresponding to a level.

    Parameters:
    -----------
    key : np.uint64
        Morton key.

    Returns:
    --------
    np.uint64
    """
    return key & np.uint64(0x7fff)


def find_bounds(sources, targets):
    """
    Find the bounds of the Octree domain describing a set of sources and targets.

    Parameters:
    -----------
    sources : np.array(shape=(N, 3), dtype=np.float32)
    targets : np.array(shape=(N, 3), dtype=np.float32)

    Returns:
    --------
    (np.array(shape=(3,), dtype=np.float32),
        np.array(shape=(3,), dtype=np.float32))
    """

    min_bound = np.min(
        np.vstack([np.min(sources, axis=0), np.min(targets, axis=0)]), axis=0
    )

    max_bound = np.max(
        np.vstack([np.max(sources, axis=0), np.max(targets, axis=0)]), axis=0
    )

    return max_bound, min_bound


@numba.njit(cache=True)
def find_center(max_bound, min_bound):
    """
    Find center of an Octree domain described by a minimum and maximum bound.

    Parameters:
    -----------
    max_bound : np.array(shape=(3,), dtype=np.float32)
    min_bound : np.array(shape=(3,), dtype=np.float32)

    Returns:
    np.array(shape=(3,), dtype=np.float32)
    """
    center = (min_bound + max_bound) / 2
    return center


def find_radius(center, max_bound, min_bound):
    """
    Find the half side length `radius' of an Octree's root node.

    Parameters:
    -----------
    center : np.array(shape=(3,), dtype=np.float32)
    max_bound : np.array(shape=(3,), dtype=np.float32)
    min_bound : np.array(shape=(3,), dtype=np.float32)

    Returns:
    --------
    np.float32
    """
    factor = 1 + 1e-5
    radius = np.max([np.max(center - min_bound), np.max(max_bound - center)]) * factor

    return radius


@numba.njit(cache=True)
def point_to_anchor(point, level, x0, r0):
    """
    Find the anchor of the octant in which a 3D Cartesian point lies.

    Parameters:
    -----------
    point : np.array(shape=(3,), dtype=np.float32)
    level : np.uint16
        Octree level of point.
    x0 : np.array(shape=(3,), dtype=np.float32)
        Center of root node of Octree.
    r0 : np.float32
        Half side length of root node.

    Returns:
    --------
    np.array(shape=(4,), dtype=np.uint16)
    """
    anchor = np.empty(4, dtype=np.uint16)
    anchor[3] = level

    xmin = x0 - r0

    side_length = 2 * r0 / (1 << level)
    anchor[:3] = np.floor((point - xmin) / side_length).astype(np.uint16)

    return anchor


@numba.njit(cache=True)
def encode_point(point, max_level, level, x0, r0):
    """
    Apply Morton encoding to a point.

    Parameters:
    -----------
    point : np.array(shape=(3,), dtype=np.float32)
    max_level : np.uint16
    level : np.uint16
    x0 : np.array(shape=(3,), dtype=np.float32)
        Center of root node of Octree.
    r0 : np.float32
        Half side length of root node.

    Returns:
    --------
    np.uint64
        Morton key.
    """
    anchor = point_to_anchor(point, level, x0, r0)
    return encode_anchor(anchor)


@numba.njit(cache=True)
def encode_points(points, level, x0, r0):
    """
    Apply morton encoding to a set of points.

    Parameters:
    -----------
    points : np.array(shape=(3, N), dtype=np.float32)
    level : np.uint16
        Octree level of point.
    x0 : np.array(shape=(3,), dtype=np.float32)
        Center of root node of Octree.
    r0 : np.float32
        Half side length of root node.

    Returns:
    --------
    np.array(shape=(N,), dtype=np.uint64)
    """
    npoints = len(points)
    keys = np.empty(npoints, dtype=np.uint64)

    anchors = np.empty((npoints, 4), dtype=np.uint16)
    anchors[:, 3] = level

    xmin = x0 - r0
    diameter = 2 * r0 / (1 << level)

    anchors[:, :3] = np.floor((points - xmin) / diameter).astype(np.uint16)

    for i in range(npoints):
        keys[i] = encode_anchor(anchors[i, :])

    return keys


# @numba.njit
def encode_anchor(anchor):
    """
    Morton encode a set of anchor coordinates and their octree level. Assume a
        maximum of 16 bits for each anchor coordinate, and 15 bits for level.
        The strategy is to examine each coordinate byte by byte, from most to
        least significant bytes, and find interleaving using the lookup table.
        Finally, level information is appended to the tail.

    Parameters:
    -----------
    anchor : np.array(shape=(4,), dtype=np.int16)

    Returns:
    --------
    key : np.uint64
    """
    x = anchor[0]
    y = anchor[1]
    z = anchor[2]
    level = anchor[3]

    key = np.uint64(0)

    # Find interleaving
    key = Z_LOOKUP[(z >> 8) & 0xff] | Y_LOOKUP[(y >> 8) & 0xff] | X_LOOKUP[(x >> 8) & 0xff]
    key = (key << 24) | Z_LOOKUP[z & 0xff] | Y_LOOKUP[y & 0xff] | X_LOOKUP[x & 0xff]

    # Append level
    key = key << 15
    key = key | level

    return key


def encode_anchors(anchors):

    keys = []

    for anchor in anchors:
        keys.append(encode_anchor(anchor))

    return np.array(keys, dtype=np.int64)


# @numba.njit
def decode_key(key):
    """
    Decode a Morton encoded key, return an anchor. The strategy is to examine
    the 64 bit key 3 bytes at a time, and extract 8 of the x, y and z bits
    from this chunk of 3 bytes. This chunking is iterative, and will therefore
    be significantly slower than the lookup based encoding.

    Paramters:
    ----------
    key : np.int64

    Returns:
    --------
    np.array(shape=(4,), np.int16)
    """
    x = 0
    y = 0
    z = 0
    level = find_level(key)
    key = key >> np.uint64(15)

    def extract(x):
        """extract every third bit from 24 bit integer"""
        ans = 0
        i = 0
        while x > 0:
            ans = ans | ((x & 1) << i)
            i += 1
            x = x >> 3
        return ans

    x = extract(key)
    x =  x | (extract((key >> 24)) << 8)

    y = extract(key >> 1)
    y = y | (extract((key >> 25)) << 8)

    z = extract(key >> 2)
    z = z | (extract((key >> 26)) << 8)

    anchor = np.array([x, y, z, level], dtype=np.int16)
    return anchor


@numba.njit
def not_ancestor(a, b):
    """
    Check if octant a is not an ancestor of octant b.

    Parameters:
    -----------
    a : int
        Morton key
    b : int
        Morton key

    Returns:
    --------
    bool
    """

    # Extract level
    level_a = find_level(a)
    level_b = find_level(b)

    if (level_a == level_b) and (a != b):
        return True

    if (level_a > level_b):
        return True

    # Remove level offset
    a = a >> 15
    b = b >> 15

    # Check remaining bits of a against b
    b = b >> (3*(level_b - level_a))

    return bool(a^b)


# @numba.njit
def find_siblings(a):
    """
    Find the siblings of a

    Parameters:
    -----------
    a : np.uint64
        Morton key

    Returns:
    --------
    [int]
    """

    suffixes = np.array([
        0, 1, 2, 3, 4, 5, 6, 7
    ], dtype=np.uint64)

    # Extract and remove level bits
    level = find_level(a)
    a = a >> np.uint64(15)

    # Clear suffix of a
    a_root = (a >> np.uint64(3)) << np.uint64(3)

    siblings = []

    for suffix in suffixes:
        sibling = a_root | suffix
        sibling = ((sibling << np.uint64(15)) | level)
        siblings.append(sibling)

    return siblings


@numba.njit
def not_sibling(a, b):
    """
    Check if octant a is not a sibling of octant b.

    Parameters:
    -----------
    a : int
        Morton key
    b : int
        Morton key

    Returns:
    --------
    bool
    """

    # Check if a and b share a parent
    a = a >> 3
    b = b >> 3

    # Get parent bits
    mask = 7
    a_parent = a & mask
    b_parent = b & mask

    return bool(a_parent^b_parent)


# @numba.njit
def find_neighbours(a):
    """
    Find all potential neighbours of octant a at a given level

    Parameters:
    -----------
    a : int
        Morton key

    Returns:
    --------
    [int]
    """
    anchor = decode_key(a)
    x = anchor[0]
    y = anchor[1]
    z = anchor[2]
    level = anchor[-1]
    max_index = 1 << level

    neighbours = []
    for i in range(-1, 2):
        for j in range(-1, 2):
            for k in range(-1, 2):

                neighbour_anchor = np.array([x+i, y+j, z+k, level])

                if ((np.any(neighbour_anchor < 0)) or (np.any(neighbour_anchor >= max_index))):
                    pass
                else:
                    neighbours.append(neighbour_anchor)

    neighbours = encode_anchors(neighbours)
    neighbours = neighbours[neighbours != a]
    return neighbours


# @numba.njit
def find_parent(a):
    """
    Find parent octant of a

    Parameters:
    -----------
    a : int
        Morton key

    Returns:
    --------
    int
    """
    # Extract and remove level bits

    n_level_bits = np.uint64(15)
    parent_shift = np.uint(3)

    level = find_level(a)
    a = a >> n_level_bits

    parent_level = np.uint64(level - 1)

    print("HERE", a, type(a), level, type(parent_level))

    parent = a >> parent_shift
    parent = parent << n_level_bits
    parent = parent | parent_level
    return parent
