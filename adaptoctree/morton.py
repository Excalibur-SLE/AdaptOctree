"""
Construct z order locational code to describe a set of points.
"""
import numba
import numpy as np


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


if __name__ == "__main__":
    res = morton_encode([0, 0, 0, 2])

    print(res)
    print(bin(res))