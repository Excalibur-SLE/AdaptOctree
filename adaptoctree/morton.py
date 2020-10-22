"""
Construct z order locational code to describe a set of points.
"""
import numba
import numpy as np





@numba.njit(cache=True)
def get_center_from_4d_index(index, x0, r0):
    pass


@numba.njit(cache=True)
def get_center_from_key(key, x0, r0):
    pass


@numba.njit(cache=True)
def find_level(key):
    pass


@numba.njit(cache=True)
def encode_points(points, level, x0, r0):
    """
    """
    npoints = len(points)
    keys = np.empty(npoints, dtype=np.int64)
    indices = np.empty((npoints, 4), dtype=np.int64)
    indices[:, -1] = level
    xmin = x0 - r0
    side_length = 2 * r0 / (1 << level)
    indices[:, :3] = np.floor((points - xmin) / side_length).astype(np.int32)

    for i in range(npoints):
        keys[i] = encode_anchor(indices[i, :])
    return keys


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
def find_anchor_from_point(point, level, x0, r0):
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
    anchor = find_anchor_from_point(point, level, x0, r0)
    return morton_encode(anchor, max_level)


def deinterleave(interleaved):
    """
    """
    x = 0
    y = 0
    z = 0

    i = 0

    while interleaved > 0:
        tmp = interleaved & 7
        print(bin(interleaved), bin(tmp))

        xi = tmp & 4
        yi = tmp & 2
        zi = tmp & 1

        interleaved = interleaved >> 3

    return np.array([xi, yi, zi])


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
def encode_anchor(anchor):
    """
    Apply Morton encoding
    """

    def split(x):
        """
        Insert two 0 bits after each of the 10 low bits of x using magic numbers.
        """
        x &= 0x000003ff;                  # x = ---- ---- ---- ---- ---- --98 7654 3210
        x = (x ^ (x << 16)) & 0xff0000ff # x = ---- --98 ---- ---- ---- ---- 7654 3210
        x = (x ^ (x <<  8)) & 0x0300f00f # x = ---- --98 ---- ---- 7654 ---- ---- 3210
        x = (x ^ (x <<  4)) & 0x030c30c3 # x = ---- --98 ---- 76-- --54 ---- 32-- --10
        x = (x ^ (x <<  2)) & 0x09249249 # x = ---- 9--8 --7- -6-- 5--4 --3- -2-- 1--0

        return x

    # Interleave bits
    key = (split(anchor[0]) << 2) + (split(anchor[1]) << 1) + (split(anchor[2]))

    # Append level
    key = key << 4
    key = key | (anchor[3])

    return key

@numba.njit
def decode_key(key):

    def remove_level(key):
        return key >> 4

    def get_level(key):
        level = key & (15)
        return level

    def compact(key):
        # Remove level
        key = remove_level(key)

        key &= 0x09249249                  # x = ---- 9--8 --7- -6-- 5--4 --3- -2-- 1--0
        key = (key ^ (key >>  2)) & 0x030c30c3 # x = ---- --98 ---- 76-- --54 ---- 32-- --10
        key = (key ^ (key >>  4)) & 0x0300f00f # x = ---- --98 ---- ---- 7654 ---- ---- 3210
        key = (key ^ (key >>  8)) & 0xff0000ff # x = ---- --98 ---- ---- ---- ---- 7654 3210
        key = (key ^ (key >> 16)) & 0x000003ff # x = ---- ---- ---- ---- ---- --98 7654 3210
        return key

    level = get_level(key)
    x = compact(key >> 2)
    y = compact(key >> 1)
    z = compact(key)

    anchor = np.array([x, y, z, level])

    return anchor


if __name__ == "__main__":
    anchor = [1, 13, 1, 2]
    print([bin(i) for i in anchor])
    print(encode_anchor(anchor))
    key = encode_anchor(anchor)
    print(str(bin(key)))
    print(key)
    print()
    print(decode_key(key))