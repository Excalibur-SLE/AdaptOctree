"""
Utility functions to construct Morton encodings from a set of points distributed
in 3D.
"""
import numba
import numpy as np


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
    Pop the last four bits, corresponding to key.
    """
    return key & 0xF


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


@numba.njit
def encode_anchor(anchor):
    """
    Apply Morton encoding
    """

    def split(x):
        """
        Insert two 0 bits after each of the 10 low bits of x using magic numbers.
        """
        x &= 0x000003ff;                 # x = ---- ---- ---- ---- ---- --98 7654 3210
        x = (x ^ (x << 16)) & 0xff0000ff # x = ---- --98 ---- ---- ---- ---- 7654 3210
        x = (x ^ (x <<  8)) & 0x0300f00f # x = ---- --98 ---- ---- 7654 ---- ---- 3210
        x = (x ^ (x <<  4)) & 0x030c30c3 # x = ---- --98 ---- 76-- --54 ---- 32-- --10
        x = (x ^ (x <<  2)) & 0x09249249 # x = ---- 9--8 --7- -6-- 5--4 --3- -2-- 1--0

        return x

    # Interleave bits
    key = (split(anchor[0]) << 2) + (split(anchor[1]) << 1) + (split(anchor[2]))

    # Append level to final 4 bits
    key = key << 4
    key = key | anchor[3]

    return key


@numba.njit
def decode_key(key):
    """
    Decode Morton key, to anchor point.
    """
    def remove_level(key):
        return key >> 4

    def find_level(key):
        level = key & 0xF
        return level

    def compact(key):
        key = remove_level(key)

        key &= 0x09249249                      # x = ---- 9--8 --7- -6-- 5--4 --3- -2-- 1--0
        key = (key ^ (key >>  2)) & 0x030c30c3 # x = ---- --98 ---- 76-- --54 ---- 32-- --10
        key = (key ^ (key >>  4)) & 0x0300f00f # x = ---- --98 ---- ---- 7654 ---- ---- 3210
        key = (key ^ (key >>  8)) & 0xff0000ff # x = ---- --98 ---- ---- ---- ---- 7654 3210
        key = (key ^ (key >> 16)) & 0x000003ff # x = ---- ---- ---- ---- ---- --98 7654 3210

        return key

    level = find_level(key)
    x = compact(key >> 2)
    y = compact(key >> 1)
    z = compact(key)

    anchor = np.array([x, y, z, level])

    return anchor


def encode_point(point, max_level, level, x0, r0):
    """
    """
    anchor = point_to_anchor(point, level, x0, r0)
    return encode_anchor(anchor)


@numba.njit(cache=True)
def encode_points(points, level, x0, r0):
    """
    """
    npoints = len(points)
    keys = np.empty(npoints, dtype=np.int64)

    anchors = np.empty((npoints, 4), dtype=np.int32)
    anchors[:, -1] = level

    xmin = x0 - r0
    diameter = 2 * r0 / (1 << level)

    anchors[:, :3] = np.floor((points - xmin) / diameter).astype(np.int32)

    for i in range(npoints):
        keys[i] = encode_anchor(anchors[i, :])

    return keys

if __name__ == "__main__":
    anchor = [1, 13, 1, 2]
    print([bin(i) for i in anchor])
    print(encode_anchor(anchor))
    key = encode_anchor(anchor)
    print(str(bin(key)))
    print(key)
    print()
    print(decode_key(key))
