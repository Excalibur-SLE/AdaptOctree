"""
Test tree construction.
"""
import numpy as np
import pytest

import adaptoctree.morton as morton
import adaptoctree.tree as tree


@pytest.fixture
def points():
    return np.random.rand(1000, 3)


@pytest.fixture
def max_level():
    return 16


@pytest.fixture
def max_points():
    return 100


@pytest.fixture
def unbalanced(points, max_level, max_points):
    return tree.build(
        points=points,
        max_level=max_level,
        max_points=max_points
    )


@pytest.fixture
def balanced(unbalanced):
    tmp = tree.balance(unbalanced, tree.find_depth(unbalanced))
    return np.fromiter(tmp, np.int64, len(tmp))



@pytest.fixture
def octree_center(points):
    min_bound, max_bound = morton.find_bounds(points)
    return morton.find_center(min_bound, max_bound)


@pytest.fixture
def octree_radius(octree_center, points):
    min_bound, max_bound = morton.find_bounds(points)
    return morton.find_radius(octree_center, max_bound, min_bound)


def test_particle_constraint(
    unbalanced, balanced, max_points, points, octree_center, octree_radius
):
    """
    Test that the number of particles in a leaf box doesn't exceed the user
        specified constraint for both the balanced and unbalanced trees.
    """

    assigned_unbalanced = tree.assign_points_to_keys(
        points, unbalanced, octree_center, octree_radius
        )
    _, counts_unbalanced = np.unique(assigned_unbalanced, return_counts=True)

    assigned_balanced = tree.assign_points_to_keys(
        points, balanced, octree_center, octree_radius
        )

    _, counts_balanced = np.unique(assigned_balanced, return_counts=True)

    assert np.all(counts_unbalanced <= max_points)
    assert np.all(counts_balanced <= max_points)


def test_balance_constraint(balanced, octree_center, octree_radius):
    """
    Test that the 2:1 balance constraint is satisfied
    """

    for i in balanced:
        for j in balanced:
            if (i != j) and morton.are_neighbours(i, j, octree_center, octree_radius):
                diff = abs(morton.find_level(i) - morton.find_level(j))
                assert diff <= 1


def test_no_overlaps(balanced):
    """
    Test that the final tree doesn't contain any overlaps
    """

    for i in balanced:
        for j in balanced:
            if i != j:
                assert i not in morton.find_ancestors(j)
                assert j not in morton.find_ancestors(i)