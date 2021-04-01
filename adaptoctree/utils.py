"""
Utility functions
"""
import hashlib

import numba
import numpy as np


def deterministic_hash(array, digest_size=5):
    """
    Compute a simple deterministic hash of a general array.
    """
    h = hashlib.blake2b(digest_size=digest_size)
    h.update(array.tobytes())
    return int(h.hexdigest(), 16)


@numba.njit(cache=True)
def deterministic_checksum(coord):
    """
    Compute a simple deterministic checksum of a 3D (integer) coordinate array.
    """
    hash = 0

    for i in coord:
        if i < 0:
            hash |= np.int16(-(2*i)+1)
        else:
            hash |= np.int16(2*i)

        hash <<= 16

    return hash
