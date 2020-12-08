"""
Construct an adaptive linear octree form a set of points.
"""

import numba
import numpy as np

import adaptoctree.morton as morton


x_mask = 0b001001001001001001001001001001001001001001001001
y_mask = 0b010010010010010010010010010010010010010010010010
z_mask = 0b100100100100100100100100100100100100100100100100

yz_mask= 0b110110110110110110110110110110110110110110110110
xz_mask= 0b101101101101101101101101101101101101101101101101
xy_mask= 0b011011011011011011011011011011011011011011011011

level_mask = 0x7FFF


@numba.njit
def decrement_x(key):
    return (((key & x_mask) - 1) & x_mask) | (key & yz_mask)

@numba.njit
def decrement_y(key):
    return (((key & y_mask) - 1) & y_mask) | (key & xz_mask)

@numba.njit
def decrement_z(key):
    return (((key & z_mask) - 1) & z_mask) | (key & xy_mask)

@numba.njit
def increment_x(key):
    return (((key | yz_mask) + 1) & x_mask) | (key & yz_mask)

@numba.njit
def increment_y(key):
    return (((key | xz_mask) + 1) & y_mask) | (key & xz_mask)

@numba.njit
def increment_z(key):
    return (((key | xy_mask) + 1) & z_mask) | (key & xy_mask)



@numba.njit
def compute_neighbours(key):
    """
    Compute neighbours at the same level
    """
    level = key & 0x7fff
    key = key >> 15

    neighbours = np.array([
        decrement_x(decrement_y(decrement_z(key))),
        decrement_x(decrement_y(key)),
        decrement_x(decrement_z(key)),
        decrement_x(key),
        decrement_x(increment_z(key)),
        decrement_x(increment_y(decrement_z(key))),
        decrement_x(increment_y(key)),
        decrement_x(increment_y(increment_z(key))),
        decrement_y(decrement_z(key)),
        decrement_y(key),
        decrement_y(increment_z(key)),
        decrement_z(key),
        increment_z(key),
        increment_y(decrement_z(key)),
        increment_y(key),
        increment_y(increment_z(key)),
        increment_x(decrement_y(decrement_z(key))),
        increment_x(decrement_y(decrement_z(key))),
        increment_x(decrement_y(key)),
        increment_x(decrement_y(increment_z(key))),
        increment_z(decrement_z(key)),
        increment_x(key),
        increment_x(increment_z(key)),
        increment_x(increment_y(decrement_z(key))),
        increment_x(increment_y(key)),
        increment_x(increment_y(increment_z(key)))
    ], np.int64)

    neighbours = (neighbours << 15) | level

    return neighbours


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


@numba.njit
def compare_level(node_level, level):
    return node_level == level

@numba.njit
def filter_tree(tree, level):
    filtered = np.empty_like(tree)
    j = 0
    for i in range(len(tree)):
        node_level = morton.find_level(tree[i])
        if compare_level(node_level, level):
            filtered[j] = tree[i]
            j += 1
    return filtered[:j]


@numba.njit
def intersect1d(ar1, ar2):

    ar1 = np.unique(ar1)
    ar2 = np.unique(ar2)

    aux = np.concatenate((ar1, ar2))
    aux.sort()
    mask = aux[1:] == aux[:-1]
    int1d = aux[:-1][mask]

    return int1d




@profile
# @numba.njit
def balance(unbalanced_tree, depth, max_level):
    """
    Single-node sequential tree balancing.

    Parameters:
    -----------
    octree : Octree
    depth : int

    Returns:
    --------
    Octree
    """

    work_set = unbalanced_tree

    balanced = np.array([-1]) # sentinel

    for level in range(depth, 0, -1):

        # Working list, filtered to current level
        work_subset = filter_tree(work_set, level)
        # Find if neighbours of leaves at this level violate balance constraint

        for key in work_subset:
            neighbours = compute_neighbours(key)
            # neighbours = morton.find_neighbours(key)
            n_neighbours = len(neighbours)

            # Invalid neighbours are any that are more than two levels coarse than the current level
            n_invalid_neighbours = n_neighbours * (level - 2)

            if n_invalid_neighbours == 0 or level == 1:
                pass

            else:
                invalid_neighbours = np.empty(shape=(n_invalid_neighbours), dtype=np.int64)

                i = 0
                for neighbour in neighbours:

                    # This can be a numba function, process neighbours
                    for invalid_level in range(level - 2, 0, -1):

                        # remove level bits
                        invalid_neighbour = neighbour >> 15
                        # add bits for invalid level key
                        invalid_neighbour = invalid_neighbour >> (
                            3 * (level - invalid_level)
                        )

                        # add new level bits
                        invalid_neighbour = invalid_neighbour << 15
                        invalid_neighbour = invalid_neighbour | invalid_level
                        invalid_neighbours[i] = invalid_neighbour
                        i += 1

                invalid_neighbours = np.unique(invalid_neighbours)

                found = intersect1d(invalid_neighbours, work_set)

                # Check if invalid neighbours exist in working list for this node,
                # q, if so remove them and replace with valid descendents
                # Within 1 level of coarseness of q
                if found.size > 0:
                    for invalid_neighbour in invalid_neighbours:

                        # This bit can also be numba-fied
                        invalid_level = morton.find_level(invalid_neighbour)
                        valid_children = morton.find_descendents(
                            invalid_neighbour, invalid_level - (level + 1)
                        )

                        #  Filter out from work set
                        work_set = work_set[work_set != invalid_neighbour]

                        # Add valid descendents to work set
                        work_set = np.hstack((work_set, valid_children))

        if balanced.size == 1 and balanced[0] == -1:
            balanced = work_subset

        else:
            balanced = np.hstack((balanced, work_subset))


    return np.unique(balanced)


def assign_points_to_keys(points, tree, x0, r0):
    """
    Assign particle positions to Morton keys in a given tree.

    Parameters:
    -----------
    points : np.array(shape=(N, 3), dtype=np.float32)
    tree : Octree

    Returns:
    --------
    np.array(shape=(N,), dtype=np.int64)
        Column vector specifying the Morton key of the node that each point is
        associated with.
    """
    # Map Morton key to bounds that they represent.
    n_points = points.shape[0]
    n_keys = tree.shape[0]
    lower_bounds = np.zeros(shape=(n_keys, 3), dtype=np.float32)
    upper_bounds = np.zeros(shape=(n_keys, 3), dtype=np.float32)

    leaves = np.zeros(n_points, dtype=np.int64)

    # Loop over all nodes to find bounds
    for i in range(n_keys):
        key = tree[i, 0]
        bounds = morton.find_node_bounds(key, x0, r0)
        lower_bounds[i : i + 2, :] = bounds[0, :]
        upper_bounds[i : i + 2, :] = bounds[1, :]

    # Loop over points, and assign to a node from the tree by examining the bounds
    for i, point in enumerate(points):
        upper_bound_index = np.all(point < upper_bounds, axis=1)
        lower_bound_index = np.all(point >= lower_bounds, axis=1)
        leaf_index = upper_bound_index & lower_bound_index

        if np.count_nonzero(leaf_index) != 1:
            a, b = tree[:, 0][leaf_index]
        leaves[i] = tree[:, 0][leaf_index]

    return leaves


def make_moon(npoints):

    np.random.seed(1)
    x = np.linspace(0, 2 * np.pi, npoints) + np.random.rand(npoints)
    y = 0.5 * np.ones(npoints) + np.random.rand(npoints)
    z = np.sin(x) + np.random.rand(npoints)

    moon = np.array([x, y, z]).T
    return moon


def main():

    import numpy as np
    import morton
    import time

    # tree = np.array([1, 2, 3, 4, 5, 6])
    # level = 1
    # filtered = filter_np_nb(tree, level)
    # # print(morton.find_level(tree))
    # print(filtered)

    n_particles = 100
    # particles = np.random.rand(n_particles, 3)
    particles = make_moon(n_particles)

    unbalanced = build(
        particles=particles,
        max_num_particles=100
    )
    depth = max(morton.find_level(unbalanced))
    max_level = 16
    balance(np.unique(unbalanced), depth, max_level)

    n_particles = int(1e3)
    # particles = np.random.rand(n_particles, 3)
    particles = make_moon(n_particles)

    start = time.time()
    unbalanced = build(particles, max_num_particles=100)
    print("build time: ", time.time()-start)

    depth = max(morton.find_level(unbalanced))
    max_level = 16
    start = time.time()
    unbalanced = np.unique(unbalanced)
    balanced = balance(unbalanced, depth, max_level)
    print("balance time: ", time.time()-start)
    print()
    print("Unbalanced tree", unbalanced.shape)
    print("Balanced tree", balanced.shape)


if __name__ == "__main__":
    main()