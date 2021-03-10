"""
Utility functions
"""
import hashlib


def deterministic_hash(array, digest_size=5):
    """
    Compute a simple deterministic hash of an array
    """
    h = hashlib.blake2b(digest_size=digest_size)
    h.update(array.tobytes())
    return int(h.hexdigest(), 16)
