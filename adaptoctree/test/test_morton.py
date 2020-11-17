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
            np.array([1, 1 , 1, 1], dtype=np.int32),
            np.array([0, 0, 0], dtype=np.float32),
            np.float32(2),
            np.array([1, 1, 1], dtype=np.float32)
        )
    ]
)
def test_find_center_from_anchor(anchor, x0, r0, expected):

    result = morton.find_center_from_anchor(anchor, x0, r0)
    assert np.array_equal(result, expected)
    assert isinstance(result[0], np.float32)


@pytest.mark.parametrize(
    "key, x0, r0, expected",
    [
        (
            np.int64(0),
            np.array([1, 1, 1], dtype=np.float32),
            np.float32(2),
            np.array([1, 1, 1], dtype=np.float32)
        )
    ]
)
def test_find_center_from_key(key, x0, r0, expected):

    result = morton.find_center_from_key(key, x0, r0)
    assert np.array_equal(result, expected)


@pytest.mark.parametrize(
    "key, expected",
    [
        (np.int64(0), np.int64(0)),
        (np.int64(32769), np.int64(1)),
        (np.int64(32770), np.int64(2))
    ]
)
def test_find_level(key, expected):

    result = morton.find_level(key)
    assert result == expected
    assert isinstance(result, np.int64)


@pytest.mark.parametrize(
    "sources, targets, expected_max_bound, expected_min_bound",
    [
        (np.array([
            [1, 1, 1],
            [2, 2, 2]
        ], dtype=np.float32),
       np.array([
            [1, 1, 1],
            [2, 2, 2]
        ], dtype=np.float32),
        np.array([2, 2, 2], dtype=np.float32),
        np.array([1, 1, 1], dtype=np.float32)
        )
    ]
)
def test_find_bounds(sources, targets, expected_max_bound, expected_min_bound):

    result = morton.find_bounds(sources, targets)
    assert np.array_equal(result[0], expected_max_bound)
    assert np.array_equal(result[1], expected_min_bound)
    assert isinstance(result[0][0], np.float32)
    assert isinstance(result[1][0], np.float32)


@pytest.mark.parametrize(
    "center, max_bound, min_bound, expected",
    [
        (
            np.array([1, 1, 1], dtype=np.float32),
            np.array([2, 2, 2], dtype=np.float32),
            np.array([0, 0, 0], dtype=np.float32),
            np.float32(1+1e-5)
        )
    ]
)
def test_find_radius(center, max_bound, min_bound, expected):

    result = morton.find_radius(center, max_bound, min_bound)
    assert result == expected
    assert isinstance(result, np.float32)


@pytest.mark.parametrize(
    "point, level, x0, r0, expected",
    [
        (
            np.array([0, 0, 0], dtype=np.float32),
            np.int32(10),
            np.array([1, 1, 1], dtype=np.float32),
            np.float32(1),
            np.array([0, 0, 0, 10], dtype=np.int32)
        )
    ]
)
def test_point_to_anchor(point, level, x0, r0, expected):

    result = morton.point_to_anchor(point, level, x0, r0)
    assert np.array_equal(result, expected)
    assert isinstance(result[0], np.int32)


@pytest.mark.parametrize(
    "point, level, x0, r0, expected",
    [
        (
            np.array([0, 0, 0], dtype=np.float32),
            np.int32(1),
            np.array([1, 1, 1], dtype=np.float32),
            np.float32(1),
            np.int64(1)
        )
    ]
)
def test_encode_point(point, level, x0, r0, expected):

    result = morton.encode_point(point, level, x0, r0)
    assert result == expected
    assert isinstance(result, np.int64)


@pytest.mark.parametrize(
    "points, level, x0, r0, expected",
    [
        (
            np.array([[0, 0, 0]], dtype=np.float32),
            np.int32(1),
            np.array([1, 1, 1], dtype=np.float32),
            np.float32(1),
            np.array([1], dtype=np.int64)
        )
    ]
)
def test_encode_points(points, level, x0, r0, expected):
    result = morton.encode_points(points, level, x0, r0)
    assert np.array_equal(result, expected)
    assert isinstance(result[0], np.int64)


@pytest.mark.parametrize(
    "anchor, expected",
    [
        (
            np.array([1, 1, 1, 1], dtype=np.int32),
            np.int64(229377)
        )
    ]
)
def test_encode_anchor(anchor, expected):
    result = morton.encode_anchor(anchor)
    assert result == expected
    assert isinstance(result, np.int64)


@pytest.mark.parametrize(
    "anchors, expected",
    [
        (
            np.array([[1, 1, 1, 1]], dtype=np.int32),
            np.array([229377], dtype=np.int64)
        )
    ]
)
def test_encode_anchors(anchors, expected):
    result = morton.encode_anchors(anchors)
    assert np.array_equal(result, expected)
    assert isinstance(result[0], np.int64)


@pytest.mark.parametrize(
    "key, expected",
    [
        (1, np.array([0, 0, 0, 1], dtype=np.int16))
    ]
)
def test_decode_key(key, expected):
    result = morton.decode_key(key)
    assert np.array_equal(result, expected)
    assert isinstance(result[0], np.int16)


@pytest.mark.parametrize(
    "a, b, expected",
    [
        (0, 1, False),
        (1, 65537, True)
    ]
)
def test_not_ancestor(a, b, expected):
    result = morton.not_ancestor(a, b)
    assert result == expected


@pytest.mark.parametrize(
    "key, expected",
    [
        (np.int64(1), np.array([1, 32769, 65537, 98305, 131073, 163841, 196609, 229377]))
    ]
)
def test_find_siblings(key, expected):
    result = np.sort(morton.find_siblings(key))
    assert np.array_equal(result, expected)
    assert isinstance(result[0], np.int64)


@pytest.mark.parametrize(
    "a, b, expected",
    [
        (
           np.int64(1), np.int64(32769), False
        ),
        (
            np.int64(0), np.int64(1), True
        ),
        (
            np.int64(2), np.int64(1), True
        )
    ]
)
def test_not_sibling(a, b, expected):
    result = morton.not_sibling(a, b)
    assert result == expected


@pytest.mark.parametrize(
    "key, expected",
    [
        (np.int64(1), np.array([32769, 65537, 98305, 131073, 163841, 196609, 229377]))
    ]
)
def test_find_neighbours(key, expected):
    result = np.sort(morton.find_neighbours(key))
    assert np.array_equal(result, expected)


@pytest.mark.parametrize(
    "key, expected",
    [
        (np.int64(1), np.int64(0)),
        (np.int64(2), np.int64(1))
    ]
)
def test_find_parent(key, expected):
    result = morton.find_parent(key)
    assert result == expected
    assert isinstance(result, np.int64)


@pytest.mark.parametrize(
    "key, x0, r0, expected",
    [
        (
            0, np.array([1, 1, 1]), 1, np.array([[0, 0, 0], [2, 2, 2]])
        )
    ]
)
def test_find_node_bounds(key, x0, r0, expected):
    result = morton.find_node_bounds(key, x0, r0)

    assert np.array_equal(result, expected)


@pytest.mark.parametrize(
    "a, b, expected",
    [
        # (1, 229377, True),
        (2, 2064386, False)
    ]
)
def test_are_neighbours(a, b, expected):
    octree_center = np.array([0.5, 0.5, 0.5])
    octree_radius = 0.5

    result = morton.are_neighbours(a, b, octree_center, octree_radius)

    assert result == expected
