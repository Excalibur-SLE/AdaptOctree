"""
Construct z order locational code to describe a set of points.
"""
import numba
import numpy as np


@numba.njit
def interleave(x, y):
    """
    Interleave two 32-bit integers, bitwise.

    Parameters:
    -----------
    x : np.int32
    y : np.int32

    Returns:
    --------
    interleaved : np.int64
    """

    interleaved = np.int64(0)

    for i in range(32):
        xi = x & (1 << i)
        yi = y & (1 << i)

        interleaved |= xi << i
        interleaved |= yi << (i+1)

    return interleaved
