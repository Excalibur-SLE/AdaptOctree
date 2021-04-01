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


@pytest.mark.parametrize(
    "_tree, expected",
    [
        (np.array([1, 2, 3]), 3)
    ]
)
def test_find_depth(_tree, expected):
    result = tree.find_depth(_tree)

    assert result == expected


def test_complete_tree(balanced):
    complete = tree.complete_tree(balanced)

    for node in balanced:
        ancestors = morton.find_ancestors(node)
        for a in ancestors:
            assert a in complete


def test_find_interaction_lists(balanced):
    """
    Currently only tests interaction lists for nodes at leaf level, mainly
        checking that the constraints on their level, and adjacency are
        satisfied.
    """
    depth = tree.find_depth(balanced)
    complete = tree.complete_tree(balanced)
    u, x, v, w = tree.find_interaction_lists(balanced, complete, depth)

    for i in range(len(complete)):
        key = complete[i]

        if key in balanced:

            # Check all u list members are adjacent
            u_i = u[i][u[i] != -1]
            u_adj_idxs = morton.are_adjacent_vec(key, u_i, depth)
            assert np.all(u_adj_idxs == 1)

            # Check all x list members are at the right level, and not adjacent
            x_i = x[i][x[i] != -1]
            x_nadj_idxs = morton.are_adjacent_vec(key, x_i, depth)
            assert np.all(x_nadj_idxs == 0)

            x_levels = morton.find_level(x_i)
            assert np.all(x_levels == morton.find_level(morton.find_parent(key)))

            # Check all w list members are at right level, and not adjacent
            w_i = w[i][w[i] != -1]
            w_nadj_idxs = morton.are_adjacent_vec(key, w_i, depth)
            assert np.all(w_nadj_idxs == 0)

            w_levels = morton.find_level(w_i)
            assert np.all(w_levels == (morton.find_level(key)-1))

            # Check all v list members are at right level, and not adjacent
            v_i = v[i][v[i] != -1]
            v_nadj_idxs = morton.are_adjacent_vec(key, v_i, depth)
            assert np.all(v_nadj_idxs == 0)

            v_levels = morton.find_level(v_i)
            assert np.all(v_levels == morton.find_level(key))


def test_find_dense_v_list():

    x0 = np.array([0.5, 0.5, 0.5])
    r0 = 0.5
    depth = 3
    key = morton.encode_point(x0, depth, x0, r0)

    v_list = tree.find_dense_v_list(key, depth)

    are_adj = morton.are_adjacent_vec(key, v_list, depth)

    # Test that v list members are not adjacent to the key
    assert np.all(are_adj == 0)

    # Test that v list members are the same level as the key
    assert np.all(morton.find_level(v_list) == morton.find_level(key))

    # Test that the v list is dense
    assert len(v_list) == 189


@pytest.mark.parametrize(
    "x0, r0, depth, expected",
    [
        (
            np.array([0.5, 0.5, 0.5]),
            0.5,
            2,
            316
        ),
        (
            np.array([0.5, 0.5, 0.5]),
            0.5,
            3,
            316
        )
    ]
)
def test_find_unique_v_list_interactions(x0, r0, depth, expected):

    # Test that the v list is dense
    v, t, h = tree.find_unique_v_list_interactions(depth, x0, r0, depth)
    assert len(v) == expected

    # Test that the checksum function works as expected
    assert len(np.unique(h)) == 316