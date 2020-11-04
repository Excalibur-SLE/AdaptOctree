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
    Find the last 16 bits of a key, corresponding to a level.
    """
    return key & 0xffff


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
        maximum of 16 bits for each anchor coordinate, and 16 bits for level.
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
    key = key << 16
    key = key | level

    return key

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
    level = key & 0xffff
    key = key >> 16

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


# Neighbours for key 000 (binary)

NEIGHBOURS = [
    4, 2, 1, # Face neighbours
    6, 5, 3, # Edge neighbours
    7        # corner neighbours
]


def find_coarsest_neighbours(key):
    """
    Compute coarses possible neighbors of a given key done with bitwise
        operations.

    Parameters:
    -----------
    key : int
        Morton key
    """

    print('before', bin(key))

    level = find_level(key)

    # Maximum box index at a given level [0, 2^level)
    max_index = 2**level - 1

    # Remove level bits
    key = key >> 16

    # Get to coarsest level bits that preserve 2:1 property, i.e. one level up
    key = key >> 3

    neighbors = []

    shifts = []

    # Coarsest possible neighbours identifiable through allowed bit-flips of
    # coarsest possible bits in Morton encoding.
    print('after', bin(key))

    pass
