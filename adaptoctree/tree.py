"""
Construct an adaptive linear octree form a set of points.
"""

import numba
import numpy as np

import adaptoctree.morton as morton




@numba.njit(parallel=True)
def build(particles, maximum_level=16, max_num_particles=100, first_level=1):

    max_bound, min_bound = morton.find_bounds(particles)
    octree_center = morton.find_center(max_bound, min_bound)
    octree_radius = morton.find_radius(octree_center, max_bound, min_bound)

    morton_keys = morton.encode_points_smt(
        particles, first_level, octree_center, octree_radius
    )
    unique_indices = np.unique(morton_keys)
    n_unique_indices = len(unique_indices)
    for index in numba.prange(n_unique_indices):
        todo_indices = np.where(morton_keys == unique_indices[index])[0]
        build_implementation(
            particles,
            maximum_level,
            max_num_particles,
            octree_center,
            octree_radius,
            morton_keys,
            todo_indices,
            first_level,
        )
    return morton_keys


@numba.njit
def build_implementation(
    particles,
    maximum_level,
    max_num_particles,
    octree_center,
    octree_radius,
    morton_keys,
    todo_indices,
    first_level,
):

    level = first_level

    todo_indices_sorted = todo_indices[np.argsort(morton_keys[todo_indices])]

    while True:
        if level == (maximum_level):
            break
        todo_list = process_level(todo_indices_sorted, morton_keys, max_num_particles)
        ntodo = len(todo_list)
        if ntodo == 0:
            break
        todo_indices = np.empty(ntodo, dtype=np.int64)
        for index in range(ntodo):
            todo_indices[index] = todo_list[index]
        if len(todo_indices) == 0:
            # We are done
            break
        else:
            morton_keys[todo_indices] = morton.encode_points(
                particles[todo_indices], level + 1, octree_center, octree_radius
            )
            todo_indices_sorted = todo_indices[np.argsort(morton_keys[todo_indices])]
            level += 1
    return morton_keys


@numba.njit
def process_level(sorted_indices, morton_keys, max_num_particles):
    """Process a level."""
    count = 0
    pivot = morton_keys[sorted_indices[0]]
    nindices = len(sorted_indices)
    todo = numba.typed.List.empty_list(numba.types.int64, allocated=nindices)
    trial_set = numba.typed.List.empty_list(numba.types.int64, allocated=nindices)
    for index in range(nindices):
        if morton_keys[sorted_indices[index]] != pivot:
            if count > max_num_particles:
                todo.extend(trial_set)
            trial_set.clear()
            pivot = morton_keys[sorted_indices[index]]
            count = 0
        count += 1
        trial_set.append(sorted_indices[index])
    # The last element in the for-loop might have
    # a too large count. Need to process this as well
    if count > max_num_particles:
        todo.extend(trial_set)
    return todo


def empty_int64_set():

    _set = {numba.int64(1)}
    _set.clear()
    return _set


INT_ARRAY = numba.types.int64[:]


@profile
# @numba.njit
def balance(unbalanced, maximum_level=16):

    # Add full tree to dict for easy indexing of levels
    tree_dict = numba.typed.Dict.empty(
        key_type=numba.types.int64,
        value_type=INT_ARRAY
    )

    depth = 0

    for level in range(maximum_level):
        tree_dict[level] = np.array([-1], dtype=np.int64) # sentinel

    for node in unbalanced:
        level = morton.find_level(node)

        if level > depth:
            depth = level

        arr = np.array([node], dtype=np.int64)
        if np.array_equal(tree_dict[level], np.array([-1], dtype=np.int64)):
            tree_dict[level] = arr

        else:
            tree_dict[level] = np.hstack((tree_dict[level], arr))

    # Dump the entire unbalanced tree into a set for easy lookup
    tree_set = set()
    tree_set.update(unbalanced)

    # Iterate up through tree
    for level in range(depth, 0, -1):

        # For each node in the working set, perform neighbour search
        for node in tree_dict[level]:
            inv_neighbours = find_invalid_neighbours(node)

            if inv_neighbours[0] != -1:

                # For each invalid neighbour, check O(1) if it is in the tree
                for inv_n in inv_neighbours:

                    # If it is in the tree, remove it
                    if inv_n in tree_set:
                        tree_set.remove(inv_n)

                    # Replace it with it's valid descendents
                    inv_n_level = morton.find_level(inv_n)

                    valid_descendents = morton.find_descendents(
                        inv_n, inv_n_level-(level+1)
                    )

                    tree_set.update(valid_descendents)

                    # Update the dict tree with the descendents
                    desc_level = morton.find_level(valid_descendents[0])

                    tmp = np.hstack((valid_descendents, tree_dict[desc_level]))
                    updated = np.unique(tmp)
                    tree_dict[desc_level] = updated

        return tree_set


@numba.njit
def find_invalid_neighbours(node):

    neighbours = morton.find_neighbours(node)
    level = morton.find_level(node)
    n_neighbours = neighbours.shape[0]

    n_inv_neighbours = n_neighbours * (level - 2)
    idx = 0

    if n_inv_neighbours <= 0:
        return np.array([-1], dtype=np.int64)

    inv_neighbours = np.zeros(shape=(n_inv_neighbours), dtype=np.int64)
    for n in neighbours:
        for l in range(level-2, 0, -1):
            # Remove level bits
            inv_n = n >> 15

            # Coarsen
            inv_n = inv_n >> (3*(level-l))

            # Add level bits
            inv_n = inv_n << 15
            inv_n = inv_n | l
            inv_neighbours[idx] = inv_n
            idx += 1

    return np.unique(inv_neighbours)


# Script

import time

import numpy as np

def make_moon(npoints):

    x = np.linspace(0, 2*np.pi, npoints) + np.random.rand(npoints)
    y = 0.5*np.ones(npoints) + np.random.rand(npoints)
    z = np.sin(x) + np.random.rand(npoints)

    moon = np.array([x, y, z]).T
    return moon

N = int(1e3)
particles = make_moon(N)

unbalanced = build(particles)

start = time.time()
build(particles)
print("Build Time ", time.time()-start)

balance(unbalanced)

start = time.time()
balanced = balance(unbalanced)
print("Balance Time: ", time.time()-start)

print(balanced)
# print(find_invalid_neighbours(229379))