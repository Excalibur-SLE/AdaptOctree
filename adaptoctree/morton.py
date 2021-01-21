"""
Utility functions to construct Morton encodings from a set of points distributed
in 3D.

Conventions:
------------
1. Morton keys are stored as type np.int64
2. Anchor index coordinates are stored as type np.int32
3. By convention, we take anchor indices to have a maximum of 16 bits.
4. By convention we take the number bits to store the level in as 15 bits.
5. The reason for using signed ints over unsigned ints is technical, and related
to Python's handling of shift operators for unsigned integers.
"""
import numba
import numpy as np

from adaptoctree.morton_lookup import (
    X_LOOKUP_ENCODE, X_LOOKUP_DECODE,
    Y_LOOKUP_ENCODE, Y_LOOKUP_DECODE,
    Z_LOOKUP_ENCODE, Z_LOOKUP_DECODE
)

import adaptoctree.types as types

# Number of bits used for level information
LEVEL_DISPLACEMENT = 15

# Mask for last 15 bits
LEVEL_MASK = types.Long(0x7FFF)

# Mask for a lowest order byte
BYTE_MASK = types.Long(0xFF)
BYTE_DISPLACEMENT = types.Long(8)

# Mask for lowest order index bits
LOWEST_ORDER_MASK = types.Long(0x7)

# Mask encapsulating a byte
NINE_BIT_MASK = types.Long(0x1FF)

# Masks for coordinate bits along specific axes in a int64 Morton key
X_MASK = types.Long(0b001001001001001001001001001001001001001001001001)
Y_MASK = types.Long(0b010010010010010010010010010010010010010010010010)
Z_MASK = types.Long(0b100100100100100100100100100100100100100100100100)

YZ_MASK = types.Long(0b110110110110110110110110110110110110110110110110)
XZ_MASK = types.Long(0b101101101101101101101101101101101101101101101101)
XY_MASK = types.Long(0b011011011011011011011011011011011011011011011011)

# Maximum level available within parameters of encoding
MAXIMUM_LEVEL = types.Long(16)


def find_center_from_anchor(anchor, x0, r0):
    """
    Find center of given Octree node from it's anchor.

    Parameters:
    -----------
    anchor : np.array(shape=(4,), dtype=np.int32)
    x0 : np.array(shape=(3,), dtype=np.float32)
        Center of root node of Octree.
    r0 : np.float32
        Half side length of root node.

    Returns:
    --------
    np.array(shape=(3,), dtype=np.float32)
    """

    xmin = x0 - r0
    level = anchor[3]
    side_length = 2 * r0 / (1 << level)

    side_length = np.float64(side_length)
    anchor = anchor.astype(np.float64)

    return (anchor[:3] + 0.5) * side_length + xmin


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


@numba.jit(
    [types.Long(types.Key), types.LongArray(types.Keys)],
    cache=True
)
def find_level(key):
    """
    Find the last 15 bits of a key, corresponding to a level.
    """
    return key & LEVEL_MASK


@numba.njit(
    [types.Bounds(types.Coords)],
    cache=True
)
def find_bounds(particles):
    """
    Find the maximum and minimum bounds describing a set of particles.
    """

    min_bound = np.array(
        [np.min(particles[:, 0]), np.min(particles[:, 1]), np.min(particles[:, 2])]
    )
    max_bound = np.array(
        [np.max(particles[:, 0]), np.max(particles[:, 1]), np.max(particles[:, 2])]
    )
    return max_bound, min_bound


@numba.njit(
    [types.Coord(types.Coord, types.Coord)],
    cache=True
)
def find_center(max_bound, min_bound):
    """
    Find center of an Octree domain described by a minimum and maximum bound.
    """
    center = (min_bound + max_bound) / 2
    return center


@numba.njit(
    [types.Double(types.Coord, types.Coord, types.Coord)],
    cache=True
)
def find_radius(center, max_bound, min_bound):
    """
    Find the half side length `radius' of an Octree's root node.
    """
    rad1 = np.max(center - min_bound)
    rad2 = np.max(max_bound - center)
    vals = np.array([rad1, rad2])
    radius = np.max(vals) * (1 + 1e-10)
    return abs(radius)


@numba.njit(
    [types.Anchor(types.Coord, types.Long, types.Coord, types.Double)],
    cache=True
)
def point_to_anchor(point, level, x0, r0):
    """
    Find the anchor of the octant in which a 3D Cartesian point lies.
    """
    anchor = np.empty(4, dtype=np.int64)
    anchor[3] = level

    xmin = x0 - r0

    side_length = 2 * r0 / (1 << level)
    anchor[:3] = np.floor((point - xmin) / side_length).astype(np.int64)

    return anchor


@numba.njit(
    [types.Key(types.Anchor)]
)
def encode_anchor(anchor):
    """
    Morton encode a set of anchor coordinates and their octree level. Assume a
        maximum of 16 bits for each anchor coordinate, and 15 bits for level.
        The strategy is to examine each coordinate byte by byte, from most to
        least significant bytes, and find interleaving using the lookup table.
        Finally, level information is appended to the tail.
    """
    x = anchor[0]
    y = anchor[1]
    z = anchor[2]
    level = anchor[3]

    key = np.int64(0)

    # Find interleaving
    key = Z_LOOKUP_ENCODE[(z >> BYTE_DISPLACEMENT) & BYTE_MASK] \
        | Y_LOOKUP_ENCODE[(y >> BYTE_DISPLACEMENT) & BYTE_MASK] \
        | X_LOOKUP_ENCODE[(x >> BYTE_DISPLACEMENT) & BYTE_MASK]

    key = (key << 24) \
        | Z_LOOKUP_ENCODE[z & BYTE_MASK] \
        | Y_LOOKUP_ENCODE[y & BYTE_MASK] \
        | X_LOOKUP_ENCODE[x & BYTE_MASK]

    # Append level
    key = key << LEVEL_DISPLACEMENT
    key = key | level

    return key


@numba.njit(
    [types.Key(types.Coord, types.Long, types.Coord, types.Double)],
    cache=True
)
def encode_point(point, level, x0, r0):
    """
    Apply Morton encoding to a point.
    """
    anchor = point_to_anchor(point, level, x0, r0)
    return encode_anchor(anchor)


@numba.njit(
    [types.Keys(types.Coords, types.Long, types.Coord, types.Double)],
    cache=True
)
def encode_points(points, level, x0, r0):
    """
    Apply morton encoding to a set of points.
    """

    npoints, _ = points.shape
    keys = np.empty(npoints, dtype=np.int64)

    anchors = np.empty((npoints, 4), dtype=np.int64)
    anchors[:, 3] = level

    xmin = x0 - r0
    diameter = 2 * r0 / (1 << level)

    anchors[:, :3] = np.floor((points - xmin) / diameter).astype(np.int64)

    for i in range(npoints):
        keys[i] = encode_anchor(anchors[i, :])

    return keys


@numba.njit(
    [types.Keys(types.Coords, types.Long, types.Coord, types.Double)],
    parallel=True
)
def encode_points_smt(points, level, x0, r0):
    """
    Apply morton encoding to a set of points, using multi-threading.
    """

    npoints, _ = points.shape
    keys = np.empty(npoints, dtype=np.int64)

    anchors = np.empty((npoints, 4), dtype=np.int64)
    anchors[:, 3] = level

    xmin = x0 - r0
    diameter = 2 * r0 / (1 << level)

    anchors[:, :3] = np.floor((points - xmin) / diameter).astype(np.int64)

    for i in numba.prange(npoints):
        keys[i] = encode_anchor(anchors[i, :])

    return keys


@numba.njit(
    [types.Keys(types.Anchors)],
    cache=True
)
def encode_anchors(anchors):
    """
    Morton encode a set of anchors.
    """
    nanchors = len(anchors) // 4
    keys = np.empty(shape=(nanchors), dtype=np.int64)

    for i in range(nanchors):
        idx = 4*i
        anchor = anchors[idx:idx+4]
        keys[i] = encode_anchor(anchor)

    return keys


@numba.njit(
    [types.Anchor(types.Key)],
    cache=True
)
def decode_key(key):
    """
    Decode a Morton encoded key, return an anchor.
        The strategy is to examine the 64 bit key 3 bytes at a time, and extract
        8 of the x, y and z bits from this chunk of 3 bytes. This chunking is
        iterative, and will therefore be significantly slower than the lookup
        based encoding.
    """
    x = 0
    y = 0
    z = 0
    level = find_level(key)
    key = key >> LEVEL_DISPLACEMENT

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
    x =  x | (extract((key >> 24)) << BYTE_DISPLACEMENT)

    y = extract(key >> 1)
    y = y | (extract((key >> 25)) << BYTE_DISPLACEMENT)

    z = extract(key >> 2)
    z = z | (extract((key >> 26)) << BYTE_DISPLACEMENT)

    anchor = np.array([x, y, z, level], dtype=np.int64)
    return anchor


@numba.njit(
    [types.Long(types.Key, types.LongArray, types.Long)],
    cache=True
)
def decode_key_lut_helper(key, lookup_table, start_shift):
    """
    Helper method for Morton key decode method that uses lookup tables.
    """
    n_loops = 7 # 8 bytes in 64 bit key
    coord = 0

    for i in range(n_loops):
        coord |= lookup_table[(key >> ((i * 9) + start_shift)) & NINE_BIT_MASK] << (3*i)

    return coord


@numba.njit(
    [types.Anchor(types.Key, types.LongArray, types.LongArray, types.LongArray)],
    cache=True
)
def decode_key_lut(key, x_lookup_decode, y_lookup_decode, z_lookup_decode):
    """
    Decode a Morton key, return an anchor, using the provided lookup tables.
    """
    level = find_level(key)
    key = key >> LEVEL_DISPLACEMENT

    x = decode_key_lut_helper(key, x_lookup_decode, 0)
    y = decode_key_lut_helper(key, y_lookup_decode, 0)
    z = decode_key_lut_helper(key, z_lookup_decode, 0)

    return np.array([x, y, z, level], np.int64)


@numba.njit(
    [types.Key(types.Key), types.Keys(types.Keys)],
    cache=True
)
def find_parent(key):
    """
    Find parent of an octant.
    """
    # Extract and remove level bits
    level = find_level(key)
    key = key >> LEVEL_DISPLACEMENT

    parent_level = level - 1

    parent = key >> 3
    parent = parent << LEVEL_DISPLACEMENT
    parent = parent | parent_level

    return parent


@numba.njit(
    types.Keys(types.Key),
    cache=True
)
def find_siblings(key):
    """
    Find the siblings of key.
    """

    suffixes = np.array([
        0, 1, 2, 3, 4, 5, 6, 7
    ], dtype=np.int64)

    # Extract and remove level bits
    level = find_level(key)
    key = key >> LEVEL_DISPLACEMENT

    # Clear suffix of the key
    root = (key >> 3) << 3

    siblings = np.empty_like(suffixes)

    for i, suffix in enumerate(suffixes):
        sibling = root | suffix
        sibling = ((sibling << LEVEL_DISPLACEMENT) | level)
        siblings[i] = sibling

    return siblings


@numba.njit(
    [types.Keys(types.Key)],
    cache=True
)
def find_children(key):
    """
    Find children of key
    """
    # Remove level bits
    level = find_level(key)
    key = key >> LEVEL_DISPLACEMENT

    # Find first child
    child = key << 3
    child = child << LEVEL_DISPLACEMENT
    child = child | (level + 1)

    # Return siblings of first child
    return find_siblings(child)


@numba.njit(
    [types.Keys(types.Key, types.Long)],
    cache=True
)
def find_descendents(key, N):
    """
    Find all descendents N levels down tree from key.
    """
    descendents = find_children(key)

    previous_left_idx = 0

    for i in range(N - 1):
        left_idx = previous_left_idx
        right_idx = 8 ** (i + 1) + left_idx
        for d in descendents[left_idx:right_idx]:
            descendents = np.hstack((descendents, find_children(d)))

        previous_left_idx = right_idx

    return descendents[previous_left_idx:]


@numba.njit(
    [types.Key(types.Key)],
    cache=True
)
def decrement_x(key):
    """
    Decrement a Morton Key in the x direction.
    """
    return (((key & X_MASK) - 1) & X_MASK) | (key & YZ_MASK)


@numba.njit(
    [types.Key(types.Key)],
    cache=True
)
def decrement_y(key):
    """
    Decrement a Morton Key in the y direction.
    """
    return (((key & Y_MASK) - 1) & Y_MASK) | (key & XZ_MASK)


@numba.njit(
    [types.Key(types.Key)],
    cache=True
)
def decrement_z(key):
    """
    Decrement a Morton Key in the z direction.
    """
    return (((key & Z_MASK) - 1) & Z_MASK) | (key & XY_MASK)


@numba.njit(
    [types.Key(types.Key)],
    cache=True
)
def increment_x(key):
    """
    Increment a Morton Key in the x direction.
    """

    return (((key | YZ_MASK) + 1) & X_MASK) | (key & YZ_MASK)


@numba.njit(
    [types.Key(types.Key)],
    cache=True
)
def increment_y(key):
    """
    Increment a Morton Key in the y direction.
    """
    return (((key | XZ_MASK) + 1) & Y_MASK) | (key & XZ_MASK)


@numba.njit(
    [types.Key(types.Key)],
    cache=True
)
def increment_z(key):
    """
    Increment a Morton Key in the z direction.
    """
    return (((key | XY_MASK) + 1) & Z_MASK) | (key & XY_MASK)


@numba.njit(
    [types.Keys(types.Key)],
    cache=True
)
def find_neighbours(key):
    """
    Compute all neighbours at the same level, even if node at a boundary.
    """
    level = key & LEVEL_MASK
    key = key >> LEVEL_DISPLACEMENT

    neighbours = np.array([
        decrement_x(decrement_y(decrement_z(key))),
        decrement_x(decrement_y(key)),
        decrement_x(decrement_y(increment_z(key))),
        decrement_x(decrement_z(key)),
        decrement_x(key),
        decrement_x(increment_z(key)),
        decrement_x(increment_y(decrement_z(key))),
        decrement_x(increment_y(key)),
        decrement_x(increment_y(increment_z(key))),
        decrement_y(decrement_z(key)),
        decrement_y(key),
        decrement_y(increment_z(key)),
        decrement_z(key),
        increment_z(key),
        increment_y(decrement_z(key)),
        increment_y(key),
        increment_y(increment_z(key)),
        increment_x(decrement_y(decrement_z(key))),
        increment_x(decrement_y(key)),
        increment_x(decrement_y(increment_z(key))),
        increment_x(decrement_z(key)),
        increment_x(key),
        increment_x(increment_z(key)),
        increment_x(increment_y(decrement_z(key))),
        increment_x(increment_y(key)),
        increment_x(increment_y(increment_z(key)))
    ], np.int64)

    # Filter out neighbours if they outside root node domain
    bound = (2 << (level-1)) - 1 #NOTE: Only works if level > 0
    x_bound = np.array([bound, 0, 0, 0])
    y_bound = np.array([0, bound, 0, 0])
    z_bound = np.array([0, 0, bound, 0])
    origin = np.array([0, 0, 0, 0])
    anchors = np.hstack((origin, x_bound, y_bound, z_bound))
    bounds = encode_anchors(anchors)
    bounds = bounds >> LEVEL_DISPLACEMENT

    mask = np.zeros_like(neighbours, dtype=np.bool_)

    for idx, neighbour in enumerate(neighbours):
        if (bounds[0] <= (neighbour & X_MASK) <= bounds[1]) \
                and  (bounds[0] <= (neighbour & Y_MASK) <= bounds[2]) \
                and  (bounds[0] <= (neighbour & Z_MASK) <= bounds[3]):
            mask[idx] = 1

    neighbours = neighbours[mask]

    # Append level bits to neighbours
    neighbours = (neighbours << LEVEL_DISPLACEMENT) | level

    return neighbours


@numba.njit(
    [types.KeySet(types.Key)],
    cache=True
)
def find_ancestors(key):
    """
    Find Morton key of all ancestors, including the key itself.
    """
    level = find_level(key)
    key = key >> LEVEL_DISPLACEMENT

    ancestors = np.zeros(shape=(level+1), dtype=np.int64)
    idx = 0

    while level > -1:

        ancestors[idx] = ((key) << LEVEL_DISPLACEMENT) | level
        key = key >> 3
        idx += 1
        level -= 1

    return set(ancestors[:idx])


def larger_than(a, b):
    """
    Check if Morton key 'a' is larger than 'b'.

    Parameters:
    -----------
    a : np.int64
    b : np.int64

    Returns:
    --------
    np.int64
        1 if True, 0 if False.
    """

    lz_a = nlz(a)
    lz_b = nlz(b)

    if lz_a > lz_b:
        return 0
    if lz_a < lz_b:
        return 1
    else:
        if a >= b:
            return 1
        return 0


@numba.njit([types.Long(types.Long), types.Int(types.Int)])
def bit_length(x):
    """
    Calculate the bit length of an integer.
    """

    n = 0
    while x > 0:
        x >>= 1
        n += 1
    return n


@numba.njit([types.Long(types.Key)])
def nlz(key):
    """
    Explicitly calculate the number of leading zeroes in a Morton key.
    """
    level = find_level(key)
    expected_bits = 3*(MAXIMUM_LEVEL-level)
    key = key >> LEVEL_DISPLACEMENT
    found_bits = 3 - bit_length(key)
    nlz = expected_bits + found_bits

    return nlz


def _partition(tree, low, high):
    """
    Helper method for quicksort implementation for Morton keys.

    Parameters:
    -----------
    tree : np.array(np.int64)
        Intended for use on linear octrees.
    low : np.int64
        Low index of partition.
    high : np.int64
        High index of partition.

    Returns:
    --------
    np.int64
        Partition index.
    """
    pivot = tree[high]
    i = low - 1

    for j in range(low, high):
        if larger_than(pivot, tree[j]):
            i += 1
            tree[i], tree[j] = tree[j], tree[i]

    tree[i+1], tree[high] = tree[high], tree[i+1]

    return i+1


def quicksort(tree, low, high):
    """
    Quicksort using Morton key special comparison.

    Parameters:
    -----------
    tree : np.array(dtype=np.int64)
        Linear octree, with unique elements.
    low : np.int64
        Starting index of sort.
    high : np.int64
        Ending index of sort.
    """
    if low < high:
        p = _partition(tree, low, high)
        quicksort(tree, low, p-1)
        quicksort(tree, p+1, high)



def find_node_bounds(key, x0, r0):
    """
    Find the physical node (box) bounds, described by a given Morton key.
    Parameters:
    -----------
    key : np.int64
        Morton key.
    x0 : np.array(shape=(3,), dtype=np.float64)
        Center of root node of Octree.
    r0 : np.float64
        Half side length of root node.
    Returns:
    --------
    np.array(shape=(2, 3), dtype=np.float64),
        Bounds corresponding to (0, 0, 0) and (1, 1, 1) indices of a unit box.
    """

    center = find_center_from_key(key, x0, r0)

    level = find_level(key)
    radius = r0 / (1 << level)

    displacement = np.array([radius, radius, radius])

    lower_bound = center - displacement
    upper_bound = center + displacement

    return lower_bound, upper_bound


@numba.njit(
    [types.Long(types.Key, types.Key, types.Long)],
    cache=True
)
def are_adjacent(a, b, tree_depth):
    """
    Check if two node 'a' and 'b' are adjacent - i.e. they share a vertex, side
        or face.

        Strategy: Convert node anchors into an absolute anchor, i.e. its
        position relative to the grid in which the finest box in the tree is.
        From this, create a bounding box around the centre of 'a' with a side
        length corresponding to the side length of 'a' and 'b' added together.
        Then check whether the displacement of 'a' and 'b' lies within this box.
    """
    def anchor_to_absolute(anchor, level_diff, scaling_factor):
        """Convert a relative anchor to an absolute anchor

        Parameters:
        -----------
        anchor : np.array(shape=(4,), dtype=np.int64)
        level_diff : int
            Calculated from depth-anchor[3]
        scaling_factor : int
            The relative size of the half side length, with respect to the
            half side length of the smallest node in the tree.
        """
        if level_diff == 0:
            return anchor[:3]
        return anchor[:3]*scaling_factor

    l_a_diff = tree_depth-find_level(a)
    sf_a = 1 << l_a_diff

    l_b_diff = tree_depth - find_level(b)
    sf_b = 1 << l_b_diff

    r_a = sf_a*0.5
    r_b = sf_b*0.5

    anchor_a = decode_key(a)
    anchor_b = decode_key(b)

    a = anchor_to_absolute(anchor_a, l_a_diff, sf_a)
    b = anchor_to_absolute(anchor_b, l_b_diff, sf_b)

    c_a = a + r_a

    _min = a - r_b
    _max = c_a + r_a + r_b

    diff = r_a + b
    if (np.all(_min <= diff) and np.all(diff <= _max)):
        return 1

    return 0


@numba.njit(
    [types.LongArray(types.Key, types.Keys, types.Long)],
    cache=True
)
def are_adjacent_vec(key, key_vec, tree_depth):
    """
    Apply are_adjacent to a vector of keys.
    """
    result = np.zeros_like(key_vec)
    i = 0
    for k in key_vec:
        result[i] = are_adjacent(key, k, tree_depth)
        i += 1
    return result
