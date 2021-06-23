"""
Construct an adaptive balanced linear octree form a set of points.
"""

import numba
import numpy as np

import adaptoctree.morton as morton
import adaptoctree.types as types
import adaptoctree.utils as utils


@numba.njit(
    [types.LongIntList(types.LongArray, types.Keys, types.Long)],
    cache=True
)
def find_work_items(sorted_work_indices, keys, max_points):
    """
    Process a level, to find the work items corresponding to particles that
        occupy a certain node that exceed the maximum points per node threshold.
        This method returns the global indices of these particles.
    """
    count = 0
    pivot = keys[sorted_work_indices[0]]
    nindices = len(sorted_work_indices)

    todo = numba.typed.List.empty_list(numba.types.int64, allocated=nindices)
    trial_set = numba.typed.List.empty_list(numba.types.int64, allocated=nindices)

    for index in range(nindices):
        if keys[sorted_work_indices[index]] != pivot:
            if count > max_points:
                todo.extend(trial_set)
            trial_set.clear()
            pivot = keys[sorted_work_indices[index]]
            count = 0
        count += 1
        trial_set.append(sorted_work_indices[index])

    # The last element in the for-loop might have
    # a too large count. Need to process this as well
    if count > max_points:
        todo.extend(trial_set)

    return todo


@numba.njit(
    [
        types.Void(
            types.Coords, types.Long, types.Long, types.Coord,
            types.Double, types.Keys, types.LongArray, types.Int
        )
    ],
    cache=True
)
def build_helper(
    points, max_level, max_points, x0, r0, keys, work_indices, start_level
):
    """
    Build helper function. Works in-place on batches of points with same Morton
        key at the coarsest level 'start_level', and maintaining the max
        particles per node constraint.

        Strategy: For all particles in a given octant of the root node at the
        'start_level', calculate if any child octants violate the max_particles
        constraint - if they do, then repartition the particles into the
        grand child octants, and so on, until the constraint is satisfied.
    """

    level = start_level

    while True:
        if level == max_level:
            break

        work_list = find_work_items(work_indices, keys, max_points)

        ntodo = len(work_list)

        if ntodo == 0:
            break

        work_indices = np.empty(ntodo, dtype=np.int64)

        for index in range(ntodo):
            work_indices[index] = work_list[index]

        keys[work_indices] = morton.encode_points(
            points[work_indices], level + 1, x0, r0
        )

        work_indices = work_indices[np.argsort(keys[work_indices])]

        level += 1


def build_uniform(points, start_level, max_points):
    """
    Build a uniform octree, that satisfies the constraint of at most 'max_points'
        per node.
    """
    max_bound, min_bound = morton.find_bounds(points)
    x0 = morton.find_center(max_bound, min_bound)
    r0 = morton.find_radius(x0, max_bound, min_bound)

    level = start_level

    built = False
    while not built:
        keys = morton.encode_points_smt(points, level, x0, r0)
        _, counts = np.unique(keys, return_counts=True)
        if np.any(counts <= max_points):
            built = True
        level += 1

    return morton.find_descendents(0, level)


@numba.njit(
    [types.Keys(types.Coords, types.Long, types.Long, types.Long)],
    parallel=True, cache=True
)
def build(points, max_level, max_points, start_level):
    """
    Build an unbalanced linear Morton encoded octree, that satisfies the
        constraint of at most 'max_points' points per node.
    """

    max_bound, min_bound = morton.find_bounds(points)
    x0 = morton.find_center(max_bound, min_bound)
    r0 = morton.find_radius(x0, max_bound, min_bound)

    keys = morton.encode_points_smt(points, start_level, x0, r0)
    unique_keys = np.unique(keys)
    n_unique_keys = len(unique_keys)

    for key_idx in numba.prange(n_unique_keys):

        work_indices = np.where(keys == unique_keys[key_idx])[0]

        build_helper(
            points=points,
            max_level=max_level,
            max_points=max_points,
            x0=x0,
            r0=r0,
            keys=keys,
            work_indices=work_indices,
            start_level=start_level,
        )

    return keys


@numba.njit(
    [types.KeySet(types.KeyList, types.Long)],
    cache=True
)
def remove_overlaps(tree, depth):
    """
    Remove the overlaps in a balanced octree.

        Strategy: Traverse the octree level by level, bottom-up, and check if
        any ancestors lie in the tree. If they do, then remove them.
    """

    result = set(tree)

    for level in range(depth, 0, -1):
        work_items = [x for x in tree if morton.find_level(x) == level]

        for work_item in work_items:
            ancestors = morton.find_ancestors(work_item)
            ancestors.remove(work_item)
            for ancestor in ancestors:
                if ancestor in result:
                    result.remove(ancestor)

    return result


@numba.njit(
    [types.KeyList(types.Keys, types.Long)],
    cache=True
)
def balance_subroutine(tree, depth):
    """
    Perform balancing to enforce the 2:1 constraint between neighbouring
        nodes in linear octree.

        Strategy: Traverse the octree level by level, bottom-up, and check if
        a given node's parent lies in the tree, add it and their respective
        siblings. This enforces the 2:1 constraint.
    """
    balanced = set(tree)

    for level in range(depth, 0, -1):
        work_items = [x for x in balanced if morton.find_level(x) == level]

        for work_item in work_items:
            neighbours = morton.find_neighbours(work_item)

            for neighbour in neighbours:
                parent = morton.find_parent(neighbour)
                parent_level = level-1

                if ~(neighbour in balanced) and ~(parent in balanced):

                    balanced.add(parent)

                    if parent_level > 0:
                        siblings = morton.find_siblings(parent)
                        balanced.update(siblings)

    return numba.typed.List(balanced)


def balance(tree, depth):
    """
    Wrapper for JIT'd balance functions.

    Parameters:
    -----------
    tree : np.array(dtype=np.int64)
    depth : np.int64

    Returns:
    --------
    np.array(np.int64)
        Balanced octree
    """
    tmp = remove_overlaps(balance_subroutine(tree, depth), depth)
    return np.fromiter(tmp, np.int64)


@numba.njit(
    [types.Keys(types.Coords, types.Keys, types.Long, types.Coord, types.Double)],
    cache=True
)
def points_to_keys(points, tree, depth, x0, r0):
    """
    Assign particle positions to Morton keys in a given tree.
    """
    n_points = points.shape[0]
    leaves = -1*np.ones(n_points, dtype=np.int64)
    tree_set = set(tree)

    keys = morton.encode_points_smt(points, depth, x0, r0)

    for i, key in enumerate(keys):
        ancestors = morton.find_ancestors(key)

        ints = ancestors.intersection(tree_set)
        if ints:
            leaves[i] = next(iter(ints))

    return leaves


@numba.njit(
    [types.Long(types.Keys)],
    cache=True
)
def find_depth(tree):
    """
    Return maximum depth of a linear octree.
    """
    levels = morton.find_level(np.unique(tree))
    return np.max(levels)



@numba.njit(
    [types.KeySet(types.Keys)],
    cache=True)
def _complete_tree(leaves):
    """
    Internal jit'd function for completing a tree.
    """
    tree_set = set(leaves)

    for leaf in leaves:
        tree_set.update(morton.find_ancestors(leaf))

    return tree_set


def complete_tree(leaves):
    """
    Complete a tree defined by its leaves.

    Parmeters:
    ----------
    leaves : np.array(dtype=np.int64)

    Returns:
    --------
    np.array(dtype=np.int64)
    """
    tree_set = _complete_tree(leaves)
    return np.fromiter(tree_set, dtype=np.int64)


@numba.njit(cache=True)
def find_interaction_lists(leaves, complete, depth):
    """
    Compute all interaction lists for a complete tree. The restrictions of Numba
        lead to a definition that uses closure to compute interactions lists for
        each node in a loop.

        Strategy: Compute shared neighbour data for each list for a given node,
        compute each list, and add to pre-allocated array containing results for
        all lists.

    Parameters:
    -----------
    leaves : np.array(dtype=np.int64)
        Linear octree, represented by its leaves.
    complete: np.array(dtype=np.int64)
        A complete octree, generated from its linear representation.
    depth : np.int64
        Depth of the octree.

    Returns:
    --------
    (np.array(shape=(n_complete, n_nodes), dtype=np.int64))
        Four-tuple containing the (u, x, v, w) lists. n_complete is the length of
        the complete tree, and n_nodes are the number of nodes pre-allocated to
        each list, determined by the maximum possible number of nodes in each
        list e.g. 189 for the v list of M2L interactions. An entry of -1 in an
        interaction list can be ignored, and marks abscence of a node.
    """
    leaves_set = set(leaves)
    complete_set = set(complete)

    u = -np.ones(shape=(len(complete), 90), dtype=np.int64)
    x = -np.ones(shape=(len(complete), 20), dtype=np.int64)
    v = -np.ones(shape=(len(complete), 189), dtype=np.int64)
    w = -np.ones(shape=(len(complete), 208), dtype=np.int64)

    def find_interaction_list(
            i, leaves, leaves_set, complete_set, depth, u, x, v, w
        ):
        """
        Internal method to find interaction list for a given key.

        Parameters:
        -----------
        i : np.int64
            Index of key in the complete tree.
        leaves : np.array(dtype=np.int64)
            Linear octree, represented by its leaves.
        leaves_set : set(np.int64)
            Set containing the linear octree.
        complete_set : set(np.int64)
            Set containing the completed octree.
        depth : np.int64
            Depth of the octree.
        u : np.array(shape=(n_complete, 90))
            U list container, nearest neighbours.
        x : np.array(shape=(n_complete, 20))
            X list container, colleagues of a node's parent which are
            non-adjacent to the node - conjugate to the W list.
        v : np.array(shape=(n_complete, 189))
            V list container, children of a node's parent's colleagues which
            are not adjacent to the node.
        w : np.array(shape=(n_complete, 208))
            W list container, children of colleagues, which are not adjacent to
            the node.
        """

        def build_parent_level_leaf(
                key, leaves_set, colleagues_parents, parent_colleagues, depth
            ):
            """
            Build the portion of the interaction list that is expected at the
                parent level of a node. Contributes to the U and X lists for
                leaf keys.

            Parameters:
            ----------
            key : np.int64
                Key for the current node being considered.
            leaves_set : set(np.int64)
            colleagues_parents : np.array(np.int64)
                The (unique) parents of the node's colleagues.
            parent_colleagues : np.array(np.int64)
                The colleagues of the node's parents.
            depth : np.int64

            Returns:
            --------
            (np.int64, np.array(dtype=np.int64))
                Four-tuple (u_ptr, u, x_ptr, x), where 'u_ptr' is the length
                of the U list contributions at this level.

            """
            # U List (P2P)
            cp_in_tree = np.zeros_like(colleagues_parents)
            i = 0
            for cp in colleagues_parents:
                if cp in leaves_set:
                    cp_in_tree[i] = cp
                    i += 1

            cp_in_tree = cp_in_tree[:i]

            adj_idxs = morton.are_adjacent_vec(key, cp_in_tree, depth)
            adjacent = cp_in_tree[adj_idxs == 1]

            # X List (P2L)
            pc_in_tree = np.zeros_like(parent_colleagues)
            i = 0
            for pc in parent_colleagues:
                if pc in leaves_set:
                    pc_in_tree[i] = pc
                    i += 1

            pc_in_tree = pc_in_tree[:i]

            not_adj_idxs = morton.are_adjacent_vec(key, pc_in_tree, depth)
            not_adjacent = pc_in_tree[not_adj_idxs == 0]

            return len(adjacent), adjacent, len(not_adjacent), not_adjacent

        def build_current_level_leaf(
                key, leaves_set, complete_set, colleagues, parent_colleagues_children, depth
            ):
            """
            Build the portion of the interaction list that is expected at the
                level of a node. Contributes to the U and V lists for leaf keys.

            Parameters:
            -----------
            key : np.int64
                Key for the current node being considered.
            leaves_set : set(np.int64)
            colleagues : np.array(np.int64)
                The node's colleagues.
            parent_colleagues_children : np.array(np.int64)
                The children of a node's parent's colleagues.
            depth : np.int64

            Returns:
            --------
            (np.int64, np.array(dtype=np.int64))
                Four-tuple (u_ptr, u, v_ptr, v), where 'u_ptr' is the length
                of the U list contributions at this level.
            """
            # U List (P2P)
            c_in_tree = np.zeros_like(colleagues)
            i = 0
            for c in colleagues:
                if c in leaves_set:
                    c_in_tree[i] = c
                    i += 1

            c_in_tree = c_in_tree[:i]
            adj_idxs = morton.are_adjacent_vec(key, c_in_tree, depth)
            adjacent = c_in_tree[adj_idxs == 1]

            # V List (M2L)
            pcc_in_tree = np.zeros_like(parent_colleagues_children)
            i = 0
            for pcc in parent_colleagues_children:
                if pcc in complete_set:
                    pcc_in_tree[i] = pcc
                    i += 1

            pcc_in_tree = pcc_in_tree[:i]
            adj_idxs = morton.are_adjacent_vec(key, pcc_in_tree, depth)
            not_adjacent = pcc_in_tree[adj_idxs == 0]

            return len(adjacent), adjacent, len(not_adjacent), not_adjacent


        def build_current_level_non_leaf(
                key, complete_set, parent_colleagues_children, depth
            ):
            """
            Build the portion of the interaction list that is expected at the
                level of a node. Contributes to the V lists for non-leaf keys.

            Parameters:
            -----------
            key : np.int64
                Key for the current node being considered.
            complete_set : set(np.int64)
            parent_colleagues_children : np.array(np.int64)
                The children of a node's parent's colleagues.
            depth : np.int64

            Returns:
            --------
            (np.int64, np.array(dtype=np.int64))
                Tuple (v_ptr, v), where 'v_ptr' is the length of the V list
                contributions at this level.
            """
            # V List (M2L)
            pcc_in_tree = np.zeros_like(parent_colleagues_children)
            i = 0
            for pcc in parent_colleagues_children:
                if pcc in complete_set:
                    pcc_in_tree[i] = pcc
                    i += 1

            pcc_in_tree = pcc_in_tree[:i]
            adj_idxs = morton.are_adjacent_vec(key, pcc_in_tree, depth)
            not_adjacent = pcc_in_tree[adj_idxs == 0]

            return len(not_adjacent), not_adjacent


        def build_child_level_leaf(key, leaves_set, colleagues_children, depth):
            """
            Build the portion of the interaction list that is expected at the
                level of a node's children. Contributes to the U and W lists for
                leaf keys.

            Parameters:
            -----------
            key : np.int64
                Key for the current node being considered.
            leaves_set : set(np.int64)
            colleagues_children : np.array(np.int64)
                The children of a node's colleagues.
            depth : np.int64

            Returns:
            --------
            (np.int64, np.array(dtype=np.int64))
                Four-tuple (u_ptr, u, w_ptr, w), where 'u_ptr' is the length
                of the U list contributions at this level.
            """
            # U List (P2P)
            i = 0
            cc_in_tree = np.zeros_like(colleagues_children)
            for cc in colleagues_children:
                if cc in leaves_set:
                    cc_in_tree[i] = cc
                    i += 1

            cc_in_tree = cc_in_tree[:i]
            adj_idxs = morton.are_adjacent_vec(key, cc_in_tree, depth)
            adjacent = cc_in_tree[adj_idxs == 1]

            # W List (M2P)
            not_adjacent = cc_in_tree[adj_idxs == 0]
            return len(adjacent), adjacent, len(not_adjacent), not_adjacent

        u_ptr = 0
        x_ptr = 0
        v_ptr = 0
        w_ptr = 0

        key = complete[i]

        if key in leaves_set:
            parent = morton.find_parent(key)
            colleagues = morton.find_neighbours(key)
            colleagues_children = morton.find_children_vec(colleagues).ravel()
            colleagues_parents = np.unique(morton.find_parent(colleagues))
            parent_colleagues = morton.find_neighbours(parent)
            parent_colleagues_children = morton.find_children_vec(parent_colleagues).ravel()

            pu_ptr, padj, px_ptr, pnadj = build_parent_level_leaf(
                key, leaves_set, colleagues_parents, parent_colleagues, depth
            )

            u[i][u_ptr:pu_ptr] = padj
            u_ptr = pu_ptr
            x[i][x_ptr:px_ptr] = pnadj

            cu_ptr, cadj, pcc_ptr, pcc_nadj = build_current_level_leaf(
                key, leaves_set, complete_set, colleagues, parent_colleagues_children, depth
            )
            u[i][u_ptr:u_ptr+cu_ptr] = cadj
            u_ptr = pu_ptr+cu_ptr
            v[i][v_ptr:pcc_ptr] = pcc_nadj

            ccu_ptr, ccadj, ccw_ptr, ccnadj = build_child_level_leaf(
                key, leaves_set, colleagues_children, depth
            )
            u[i][u_ptr:u_ptr+ccu_ptr] = ccadj
            w[i][w_ptr:ccw_ptr] = ccnadj

        else:
            parent = morton.find_parent(key)
            parent_colleagues = morton.find_neighbours(parent)
            parent_colleagues_children = morton.find_children_vec(parent_colleagues).ravel()
            pcc_ptr, pcc_nadj = build_current_level_non_leaf(
                key, complete_set, parent_colleagues_children, depth
            )
            v[i][v_ptr:pcc_ptr] = pcc_nadj

    for i in range(len(complete)):
        find_interaction_list(
            i, leaves, leaves_set, complete_set, depth, u, x, v, w
        )

    return u, x, v, w


@numba.njit(
    [types.Keys(types.Key, types.Long)],
    cache=True
)
def find_dense_v_list(key, depth):
    """
    Find the V list of a key if it were in a non-adaptive tree.
    """

    parent = morton.find_parent(key)
    parent_neighbours = morton.find_neighbours(parent)
    parent_neigbhours_children = morton.find_children_vec(parent_neighbours).ravel()
    are_adj = morton.are_adjacent_vec(key, parent_neigbhours_children, depth)

    return parent_neigbhours_children[are_adj == 0]


@numba.jit(
    [types.Coords(types.Coord, types.Double)],
    cache=True
)
def find_vertices(x0, r0):
    """
    Find the physical vertices of the root node of an Octree.
    """
    return np.array([
        [x0[0]+r0, x0[1]+r0, x0[2]+r0],
        [x0[0]-r0, x0[1]+r0, x0[2]+r0],
        [x0[0]+r0, x0[1]-r0, x0[2]+r0],
        [x0[0]+r0, x0[1]+r0, x0[2]-r0],
        [x0[0]-r0, x0[1]-r0, x0[2]+r0],
        [x0[0]-r0, x0[1]+r0, x0[2]-r0],
        [x0[0]+r0, x0[1]-r0, x0[2]-r0],
        [x0[0]-r0, x0[1]-r0, x0[2]-r0],
    ])


def find_unique_v_list_interactions(level, x0, r0, depth):
    """
    Find the unique V list interactions for a given level of the octree.
        There are at most 316 = 7^3-3^3 such interactions in the non-adaptive
        case.

        Strategy: Find the Morton encoding of a point at the centre of the
        domain, and compute redundant transfer vectors of all siblings and
        neighbours. Repeat this for the Morton keys of all vertices. This is
        a hack and contains a lot of redundant calculations.

    Parameters:
    -----------
    level : np.int64
        Octree level
    x0 : np.array(shape=(3,), dtype=np.float64)
        Center of octree root node.
    r0 : np.float64
        Half side length of octree root node.
    depth : np.int64
        Depth of the octree.
    digest_size : np.int64
        Size of hashed digest, default 5.

    Returns:
    --------
    (np.array(np.int64), np.array(np.int64), np.array(np.int64))
        Return a triple containing (i) The Morton keys of unique source nodes
        (ii) Their corresponding target nodes and (iii) an array of hashes
        corresponding to their transfer vectors for lookup.
    """
    # Encode the centre, and find it's neighbours
    center = morton.encode_point(x0, level, x0, r0)
    neighbours = morton.find_neighbours(center)
    neighbours = np.hstack((neighbours, np.array([center])))

    # Encode the vertices and their neighbours
    vertices = find_vertices(x0, r0)
    vertices = [morton.encode_point(v, level, x0, r0) for v in vertices]
    vertices_siblings = np.array([morton.find_siblings(v) for v in vertices]).ravel()

    # This is a hack, and contains a lot of redundant calculations
    # TODO: Optimise for the level 2 of an octree, these are all algebraic
    # relationships.
    neighbours = np.hstack((neighbours, vertices_siblings))

    redundant_v_list = []
    transfer_vectors = []

    targets = []

    # Compute all (redundant) transfer vectors for central key, and
    # its neighbours
    for neighbour in neighbours:
        v_list = find_dense_v_list(neighbour, depth)

        redundant_v_list.extend(v_list)
        targets.extend(neighbour*np.ones(len(v_list), dtype=np.int64))

        res = morton.find_transfer_vectors(neighbour, v_list)
        transfer_vectors.extend([i for i in res])

    targets = np.array(targets).ravel()
    redundant_v_list = np.array(redundant_v_list)

    unique_idxs = []
    unique_transfer_vectors = []

    # Compute the unique transfer vectors, in hashed form
    for i in range(len(transfer_vectors)):
        transfer_vector = transfer_vectors[i]

        if transfer_vector not in unique_transfer_vectors:
            unique_idxs.append(i)
            unique_transfer_vectors.append(transfer_vector)

    unique_transfer_vectors = np.array(unique_transfer_vectors)

    return redundant_v_list[unique_idxs], targets[unique_idxs], unique_transfer_vectors
