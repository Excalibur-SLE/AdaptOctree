"""
Test tree construction.
"""
import numpy as np
import pytest

import adaptoctree.morton as morton
import adaptoctree.tree as tree


@pytest.fixture
def n_particles():
    return 100


@pytest.fixture
def sources(n_particles):
    return np.random.rand(n_particles, 3)


@pytest.fixture
def maximum_level():
    return 5


@pytest.fixture
def maximum_particles():
    return 10


@pytest.fixture
def unbalanced(sources, maximum_level, maximum_particles):
    built, depth, _ = tree.build(
        sources=sources,
        targets=sources,
        maximum_level=maximum_level,
        maximum_particles=maximum_particles
    )

    return built, depth


@pytest.fixture
def balanced(unbalanced, maximum_level):
    return tree.balance(*unbalanced, maximum_level)


@pytest.fixture
def bounds(sources):
    return morton.find_bounds(sources, sources)


@pytest.fixture
def octree_center(bounds):
    return morton.find_center(*bounds)


@pytest.fixture
def octree_radius(octree_center, bounds):
    return morton.find_radius(octree_center, *bounds)


def test_particle_constraint(
        sources, unbalanced, balanced, octree_center,
        octree_radius, maximum_particles
        ):
    """
    Test that the number of particles in a leaf box doesn't exceed the user
        specified constraint for both the balanced and unbalanced trees.
    """
    assigned_unbalanced = tree.assign_points_to_keys(
        sources, unbalanced[0], octree_center, octree_radius
        )
    _, counts_unbalanced = np.unique(assigned_unbalanced, return_counts=True)

    assigned_balanced = tree.assign_points_to_keys(
        sources, balanced, octree_center, octree_radius
        )

    _, counts_balanced = np.unique(assigned_balanced, return_counts=True)

    assert np.all(counts_unbalanced < maximum_particles)
    assert np.all(counts_balanced < maximum_particles)


def test_tree_balancing(balanced, octree_center, octree_radius):

    for key_i, level_i in balanced:
        for key_j, level_j in balanced:
            if key_i != key_j:
                if morton.are_neighbours(key_i, key_j, octree_center, octree_radius):
                    assert abs(level_i-level_j) <= 1

