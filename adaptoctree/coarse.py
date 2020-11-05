"""
Coarse geometries for neighbours of a given level 2 octant.

Key corresponds to level 2 box, values correspond to a list of keys of coarsest
adjacent boxes.
"""

GEOMETRIES = {
    0x30: [
        0x0, 0x1, 0x2, 0x3,
        0x4, 0x5, 0x7, 0x36,
        0x32,
    ]
}