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
LongArray2D = numba.int64[:,:]
IntArray = numba.int32[:]
IntArray2D = numba.int32[:,:]
IntList = numba.types.ListType(Int)
LongList = numba.types.ListType(Long)
SingleCoord = numba.float32[:]
DoubleCoord = numba.float64[:]
SingleCoords = numba.float32[:,:]
DoubleCoords = numba.float64[:,:]
SingleBounds = numba.types.UniTuple(SingleCoord, 2)
DoubleBounds = numba.types.UniTuple(DoubleCoord, 2)
Void = numba.types.void
