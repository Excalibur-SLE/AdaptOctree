"""
Utility functions
"""
import hashlib


def deterministic_hash(string, digest_size=10):
    """
    Compute a simple deterministic hash.
    """
    h = hashlib.blake2b(digest_size=digest_size)
    h.update(string.encode())
    return int(h.hexdigest(), 16)
