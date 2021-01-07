"""
Test tree construction.
"""
import numpy as np
import pytest

import adaptoctree.morton as morton
import adaptoctree.tree as tree


@pytest.fixture
def points():
    return np.random.rand(1000, 3).astype(np.float64)


@pytest.fixture
def max_level():
    return np.int64(16)


@pytest.fixture
def max_points():
    return np.int64(100)


@pytest.fixture
def unbalanced(points, max_level, max_points):
    return tree.build(
        points=points,
        max_level=max_level,
        max_points=max_points,
        start_level=np.int64(0)
    )


@pytest.fixture
def balanced(unbalanced):
    tmp = tree.balance(unbalanced, tree.find_depth(unbalanced))
    return np.fromiter(tmp, np.int64, len(tmp))


@pytest.fixture
def octree_center(points):
    min_bound, max_bound = morton.find_bounds(points)
    return morton.find_center(max_bound, min_bound)


@pytest.fixture
def octree_radius(octree_center, points):
    min_bound, max_bound = morton.find_bounds(points)
    return morton.find_radius(octree_center, max_bound, min_bound)


@pytest.fixture
def assigned_unbalanced(unbalanced, points, octree_center, octree_radius):
    depth = tree.find_depth(unbalanced)
    return tree.points_to_keys(points, unbalanced, depth, octree_center, octree_radius)


@pytest.fixture
def assigned_balanced(balanced, points, octree_center, octree_radius):
    depth = tree.find_depth(balanced)
    return tree.points_to_keys(points, balanced, depth, octree_center, octree_radius)


def test_particle_constraint(max_points, assigned_unbalanced, assigned_balanced):
    """
    Test that the number of particles in a leaf box doesn't exceed the user
        specified constraint for both the balanced and unbalanced trees.
    """

    _, counts_unbalanced = np.unique(assigned_unbalanced, return_counts=True)

    _, counts_balanced = np.unique(assigned_balanced, return_counts=True)

    assert np.all(counts_unbalanced <= max_points)
    assert np.all(counts_balanced <= max_points)


def test_no_overlaps(balanced):
    """
    Test that the final tree doesn't contain any overlaps
    """

    for i in balanced:
        for j in balanced:
            if i != j:
                assert i not in morton.find_ancestors(j)
                assert j not in morton.find_ancestors(i)