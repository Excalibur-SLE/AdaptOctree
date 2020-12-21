"""
Type aliases.
"""
import numba


Key = numba.int64
Keys = numba.int64[:]
KeySet = numba.types.Set(Key)
Anchor = numba.int32[:]
Anchors = numba.int32[:]
Float = numba.float32
Int = numba.int32
Long = numba.int64
IntArray = numba.int32[:]
Coord = numba.float32[:]
Coords = numba.float32[:,:]
Bounds = numba.types.UniTuple(Coord, 2)