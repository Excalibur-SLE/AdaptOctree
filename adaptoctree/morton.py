"""
Utility functions to construct Morton encodings from a set of points distributed
in 3D.
"""
import sys

import numba
import numpy as np

from morton_lookup import (
    X_LOOKUP,
    Y_LOOKUP,
    Z_LOOKUP,
    EIGHT_BIT_MASK, TWENTY_FOUR_BIT_MASK
    )


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


def encode_anchor(anchor):
    """16 bit anchor coordinates, 12 bit level
    """
    x = anchor[0]
    y = anchor[1]
    z = anchor[2]
    level = anchor[3]

    key = 0

    # Start with 2nd byte
    key = Z_LOOKUP[(z >> 8) & 0xff] | Y_LOOKUP[(y >> 8) & 0xff] | X_LOOKUP[(x >> 8) & 0xff]
    key = (key << 24) | Z_LOOKUP[z & 0xff] | Y_LOOKUP[y & 0xff] | X_LOOKUP[x & 0xff]
    key = key << 16
    key = key | level
    return key


def decode_key(key):

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


if __name__ == "__main__":
    anchor = np.array([1, 1, 1, 1], dtype=np.int32)
    print([bin(i) for i in anchor])
    print()
    key = encode_anchor(anchor)
    print(bin(key), 'key', key)
    anchor = decode_key(key)
    print(anchor)
    # print(decode_key(key))
