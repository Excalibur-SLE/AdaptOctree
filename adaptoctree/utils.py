"""
Utility functions
"""
import hashlib


def deterministic_hash(string):
    """
    Compute a simple deterministic hash.
    """
    h = hashlib.md5(string.encode())
    return int(h.hexdigest(), 16)
