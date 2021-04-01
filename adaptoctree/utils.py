"""
Utility functions
"""
import hashlib

import numba
import numpy as np


def deterministic_hash(array, digest_size=5):
    """
    Compute a simple deterministic hash of an array
    """
    h = hashlib.blake2b(digest_size=digest_size)
    h.update(array.tobytes())
    return int(h.hexdigest(), 16)


@numba.njit(cache=True)
def simple_hash(coord, digest_size=5):
    """
    Compute a simpler deterministic hash of a 3D coordinate, that importantly
        can be jitted.
    """
    hash = 0

    for i in coord:
        if i < 0:
            hash |= np.int16(-(2*i)+1)
        else:
            hash |= np.int16(2*i)

        hash <<= 16

    return hash