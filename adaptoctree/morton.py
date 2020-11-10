"""
Utility functions to construct Morton encodings from a set of points distributed
in 3D.
"""
import numba
import numpy as np

from adaptoctree.morton_lookup import X_LOOKUP, Y_LOOKUP, Z_LOOKUP


@numba.njit(cache=True)
def find_center_from_anchor(anchor, x0, r0):
    """
    Get center of given Octree node from it's anchor

    -----------
    anchor : np.array(shape=(4,), dtype=np.int32)
    x0 : np.array(shape=(3,))
        Center of root node of Octree.
    r0 : np.float64
        Half width length of root node.
    Returns:
    --------
    np.array(shape=(3,))
    """

    xmin = x0 - r0
    level = anchor[-1]
    side_length = 2 * r0 / (1 << level)

    return (anchor[:3] + .5) * side_length + xmin


@numba.njit(cache=True)
def find_center_from_key(key, x0, r0):
    anchor = decode_key(key)
    return find_center_from_anchor(anchor, x0, r0)


@numba.njit(cache=True)
def find_level(key):
    """
    Find the last 15 bits of a key, corresponding to a level.
    """
    return key & 0x7fff


def find_bounds(sources, targets):
    """
    """

    min_bound = np.min(
        np.vstack([np.min(sources, axis=0), np.min(targets, axis=0)]), axis=0
    )

    max_bound = np.max(
        np.vstack([np.max(sources, axis=0), np.max(targets, axis=0)]), axis=0
    )

    return max_bound, min_bound


def find_center(max_bound, min_bound):
    """
    """
    center = (min_bound + max_bound) / 2
    return center


def find_radius(center, max_bound, min_bound):
    """
    """
    factor = 1 + 1e-5
    radius = np.max([np.max(center - min_bound), np.max(max_bound - center)]) * factor

    return radius


@numba.njit(cache=True)
def point_to_anchor(point, level, x0, r0):
    """
    """
    anchor = np.empty(4, dtype=np.int32)
    anchor[3] = level

    xmin = x0 - r0

    side_length = 2 * r0 / (1 << level)
    anchor[:3] = np.floor((point - xmin) / side_length).astype(np.int32)

    return anchor


@numba.njit(cache=True)
def encode_point(point, max_level, level, x0, r0):
    """
    """
    anchor = point_to_anchor(point, level, x0, r0)
    return encode_anchor(anchor)


@numba.njit(cache=True)
def encode_points(points, level, x0, r0):
    """
    Apply morton encoding to a set of points, by first finding out which
    """
    npoints = len(points)
    keys = np.empty(npoints, dtype=np.int64)

    anchors = np.empty((npoints, 4), dtype=np.int16)
    anchors[:, -1] = level

    xmin = x0 - r0
    diameter = 2 * r0 / (1 << level)

    anchors[:, :3] = np.floor((points - xmin) / diameter).astype(np.int32)

    for i in range(npoints):
        keys[i] = encode_anchor(anchors[i, :])

    return keys


@numba.njit
def encode_anchor(anchor):
    """
    Morton encode a set of anchor coordinates and their octree level. Assume a
        maximum of 15 bits for each anchor coordinate, and 16 bits for level.
        Strategy is to examine byte by byte, from most to least significant
        bytes, and find interleaving using the lookup table. Finally, level
        information is appended to the tail.

    Parameters:
    -----------
    anchor : np.array(shape=(4,), dtype=np.int16)

    Returns:
    --------
    key : np.int64
    """
    x = anchor[0]
    y = anchor[1]
    z = anchor[2]
    level = anchor[3]

    key = 0

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


@numba.njit
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
    level = key & 0x7fff
    key = key >> 15

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


def find_deepest_first_descendent(a, maximum_level):
    """
    Find Morton key of deepest first descendent of octant A

    Parameters:
    -----------
    a : int
        Morton key

    Returns:
    --------
    int
        Deepest first descendent of a given octant
    """
    # Get level
    level = find_level(a)

    # Remove level bits
    dfd = a >> 16

    # Find dfd
    while level < maximum_level:

        dfd =  dfd << 3
        level += 1

    # Append level information
    dfd = dfd << 16
    dfd |= maximum_level

    return dfd


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
    a = a >> 16
    b = b >> 16

    # Check remaining bits of a against b
    b = b >> (3*(level_b - level_a))

    return bool(a^b)


@numba.njit
def find_siblings(a):
    """
    Find the siblings of a

    Parameters:
    -----------
    a : int
        Morton key

    Returns:
    --------
    [int]
    """

    suffixes = [
        0, 1, 2, 3, 4, 5, 6, 7
    ]

    # Extract and remove level bits
    level = find_level(a)
    a = a >> 15

    # Clear suffix of a
    a_root = (a >> 3) << 3

    siblings = []

    for suffix in suffixes:
        sibling = a_root | suffix
        sibling = ((sibling << 15) | level)
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


@numba.njit
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

                if ((not np.any(neighbour_anchor < 0))
                        or (not np.any(neighbour_anchor >= max_index))):
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
    level = find_level(a)
    a = a >> 15

    parent_level = level - 1

    parent = a >> 3
    parent = parent << 15
    parent = parent | parent_level
    return parent


if __name__ == "__main__":

    anchor = np.array([1,0,1,1], dtype=np.int16)

    key = encode_anchor(anchor)

    siblings = find_siblings(key)

    print(f'key {key}, bin(key) {bin(key)}')
    print('siblings', siblings)
    print('bin siblings', [bin(s) for s in siblings])