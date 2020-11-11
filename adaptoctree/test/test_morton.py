"""
Test for Morton encoding functions.
"""

import numpy as np
import pytest

import adaptoctree.morton as morton


@pytest.mark.parametrize(
    "anchor, x0, r0, expected",
    [
        (
            np.array([1, 1 , 1, 1], dtype=np.int16),
            np.array([0, 0, 0], dtype=np.float32),
            np.float32(2),
            np.array([1, 1, 1], dtype=np.int16)
        )
    ]
)
def test_find_center_from_anchor(anchor, x0, r0, expected):

    result = morton.find_center_from_anchor(anchor, x0, r0)

    assert np.array_equal(result, expected)
    assert isinstance(result, type(expected))


def test_find_center_from_key():
    pass


@pytest.mark.parametrize(
    "key, expected",
    [
        (np.uint64(0), 0),
        (np.uint64(32769), 1),
        (np.uint64(32770), 2)
    ]
)
def test_find_level(key, expected):

    result = morton.find_level(key)
    assert result == expected
    assert isinstance(result, np.uint64)


def test_find_bounds():
    pass


def test_find_radius():
    pass


def test_point_to_anchor():
    pass


def test_encode_point():
    pass


def test_encode_points():
    pass


def test_encode_anchor():
    pass


def test_encode_anchors():
    pass


def test_decode_key():
    pass


def test_not_ancestor():
    pass


def test_find_siblings():
    pass


def test_not_sibling():
    pass


def test_find_neighbours():
    pass


def test_find_parent():
    pass
