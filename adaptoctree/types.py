"""
Type aliases.
"""
import numba


Key = numba.int64
Keys = numba.int64[:]
KeySet = numba.types.Set(Key)
KeyList = numba.types.ListType(Key)
Anchor = numba.int64[:]
Anchors = numba.int64[:]
Single = numba.float32
Double = numba.float64
Int = numba.int32
Long = numba.int64
LongArray = numba.int64[:]
IntArray = numba.int32[:]
IntList = numba.types.ListType(Int)
LongIntList = numba.types.ListType(Long)
Coord = numba.float64[:]
Coords = numba.float64[:,:]
Bounds = numba.types.UniTuple(Coord, 2)
Void = numba.types.void