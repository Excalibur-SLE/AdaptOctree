"""
Utility functions to construct Morton encodings from a set of points distributed
in 3D.
"""
import sys

import numba
import numpy as np

from morton_lookup import (
    X_LOOKUP, Y_LOOKUP, Z_LOOKUP,
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


@numba.njit(cache=True)
def encode_anchor(anchor):
    """
    Apply Morton encoding to a 3D anchor using a lookup table. Assume 64 bit key,
    with 60 bits for key, and 4 bits reserved for level. At most 20 bits for
    each anchor coordinate, stored in a 32 bit integer.

    Parameters:
    -----------
    anchor : np.array(shape=(4,), dtype=np.int32)

    Returns:
    --------
    np.int64
    """

    key = 0

    # Get least significant bits of all coordinate values
    x = anchor[0]
    y = anchor[1]
    z = anchor[2]

    level = anchor[3]

    mask = EIGHT_BIT_MASK

    while mask > 0:

        # Shift up key
        key = key << 8

        x_most_bits = x & mask
        y_most_bits = y & mask
        z_most_bits = z & mask

        # print("here", [bin(x) for x in [x_most_bits, y_most_bits, z_most_bits]])
        # print("mask", bin(mask))

        # Find splitting
        x_split = X_LOOKUP[x_most_bits]
        y_split = Y_LOOKUP[y_most_bits]
        z_split = Z_LOOKUP[z_most_bits]

        # Merge split x, y and z
        merged = x_split | y_split | z_split

        # Add to key
        key = key | merged

        # Shift mask down
        mask = mask >> 0x8

    # Append level
    key = key << 0x4
    key = key | anchor[3]

    return key


@numba.njit(cache=True)
def decode_key(key):
    """
    Decode a 3D Morton key using lookup tables, assume 64 bit keys.

    Parameters:
    -----------
    key : np.int64

    Returns:
    --------
    np.array(shape=(4,), dtype=np.int32)
    """

    # Get level
    level = key & 0xf

    # Remove level
    key = key >> 0x4

    x = y = z = 0

    mask = TWENTY_FOUR_BIT_MASK

    x_mask = X_LOOKUP[-1]
    y_mask = Y_LOOKUP[-1]
    z_mask = Z_LOOKUP[-1]

    while mask > 0:

        # Find most significant 24 bits while they exist
        most_significant_bits = mask & key

        # Extract bits for coords
        x_split = x_mask & most_significant_bits
        y_split = y_mask & most_significant_bits
        z_split = z_mask & most_significant_bits

        x = x << 0x8
        x = x | (np.int32(np.where(X_LOOKUP == x_split)[0][0]))

        y = y << 0x8
        y = y | (np.int32(np.where(Y_LOOKUP == y_split)[0][0]))

        z = z << 0x8
        z = z | (np.int32(np.where(Z_LOOKUP == z_split)[0][0]))

        mask = mask >> 0x18

    anchor = np.array([x, y, z, level], dtype=np.int32)

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
    anchor = np.array([1, 1, 1, 2], dtype=np.int32)
    print([bin(i) for i in anchor])
    print(encode_anchor(anchor))
    key = encode_anchor(anchor)
    print(str(bin(key)))
    print(key)
    print()
    print(decode_key(key))
