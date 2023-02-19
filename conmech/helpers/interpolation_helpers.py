import random
from ctypes import ArgumentError

import numba
import numpy as np

from conmech.helpers import lnh, nph
from conmech.helpers.spatial_hashing import initialize_hasher_numba, query_hasher_numba
from conmech.helpers.tmh import Timer


def decide(scale):
    return np.random.uniform(low=0, high=1) < scale


def choose(options):
    return random.choice(options)


def get_mean(dimension, scale):
    return nph.generate_normal(rows=1, columns=dimension, sigma=scale / 3)


def generate_mesh_corner_scalars(dimension: int, scale: float):
    random_vector = nph.generate_normal(rows=(2**dimension), columns=1, sigma=scale / 3)
    clipped_vector = np.maximum(-scale, np.minimum(random_vector, scale))
    normalized_cliped_vector = clipped_vector - np.mean(clipped_vector)
    return 1 + normalized_cliped_vector


def generate_corner_vectors(dimension: int, scale: float):
    corner_vectors = nph.generate_normal(rows=(2**dimension), columns=dimension, sigma=scale / 3)
    normalized_corner_vectors = corner_vectors - np.mean(corner_vectors, axis=0)
    return normalized_corner_vectors


@numba.njit
def interpolate_scaled_nodes_numba(scaled_nodes: np.ndarray, corner_vectors: np.ndarray):
    if np.min(scaled_nodes) < 0 or np.max(scaled_nodes) > 1:
        raise ArgumentError

    dimension = scaled_nodes.shape[-1]
    out_dim = corner_vectors.shape[-1]
    result = np.zeros((len(scaled_nodes), out_dim))
    if dimension == 2:
        __interpolate_scaled_nodes_2d_numba(result, scaled_nodes, corner_vectors)
    elif dimension == 3:
        __interpolate_scaled_nodes_3d_numba(result, scaled_nodes, corner_vectors)
    else:
        raise ArgumentError
    # result = interpn(points=grid, values=values, xi=scaled_nodes, method="linear")
    return result


@numba.njit
def __interpolate_scaled_nodes_2d_numba(
    result: np.ndarray, scaled_nodes: np.ndarray, corner_vectors: np.ndarray
):
    corner_values = corner_vectors.reshape(2, 2, -1)
    for i, node in enumerate(scaled_nodes):
        interpolated_values_1 = interpolate_node_numba(corner_values, node[0])
        interpolated_values_2 = interpolate_node_numba(interpolated_values_1, node[1])
        result[i] = interpolated_values_2
    return result


@numba.njit
def __interpolate_scaled_nodes_3d_numba(
    result: np.ndarray, scaled_nodes: np.ndarray, corner_vectors: np.ndarray
):
    corner_values = corner_vectors.reshape(2, 2, 2, -1)
    for i, node in enumerate(scaled_nodes):
        interpolated_values_1 = interpolate_node_numba(corner_values, node[0])
        interpolated_values_2 = interpolate_node_numba(interpolated_values_1, node[1])
        interpolated_values_3 = interpolate_node_numba(interpolated_values_2, node[2])
        result[i] = interpolated_values_3
    return result


@numba.njit(inline="always")
def interpolate_node_numba(values, scale):
    return values[0] * scale + values[1] * (1 - scale)


def get_mesh_callback(corner_vectors):
    if len(corner_vectors) == 4:
        corner_values = corner_vectors.reshape(2, 2, -1)

        def interpolate_2d(dim, tag, x, y, z, lc):
            _ = dim, tag, z, lc
            interpolated_values_1 = interpolate_node_numba(corner_values, x)
            reinterpolated_values_2 = interpolate_node_numba(interpolated_values_1, y)
            return reinterpolated_values_2

        return interpolate_2d

    if len(corner_vectors) == 8:
        corner_values = corner_vectors.reshape(2, 2, 2, -1)

        def interpolate_3d(dim, tag, x, y, z, lc):
            _ = dim, tag, lc
            interpolated_values_1 = interpolate_node_numba(corner_values, x)
            interpolated_values_2 = interpolate_node_numba(interpolated_values_1, y)
            interpolated_values_3 = interpolate_node_numba(interpolated_values_2, z)
            return interpolated_values_3

        return interpolate_3d

    raise ArgumentError


def scale_nodes_to_cube(nodes):
    nodes_min = np.min(nodes, axis=0)
    nodes_max = np.max(nodes, axis=0)
    scaled_nodes = (nodes - nodes_min) / (nodes_max - nodes_min)
    return scaled_nodes


def interpolate_3d_corner_vectors(nodes: np.ndarray, base: np.ndarray, corner_vectors: np.ndarray):
    # orthonormal matrix; inverse equals transposition
    upward_nodes = lnh.get_in_base(nodes, base.T)
    scaled_nodes = scale_nodes_to_cube(upward_nodes)
    upward_vectors_interpolation = interpolate_scaled_nodes_numba(
        scaled_nodes=scaled_nodes,
        corner_vectors=corner_vectors,
    )

    vectors_interpolation = lnh.get_in_base(upward_vectors_interpolation, base)
    # assert np.abs(np.mean(vectors_interpolation)) < 0.1
    return vectors_interpolation


def interpolate_corners(
    initial_nodes: np.ndarray,
    mean_scale: float,
    corners_scale_proportion: float,
    base: np.ndarray,
    zero_out_proportion: float = 0,
):
    if decide(zero_out_proportion):
        return np.zeros_like(initial_nodes)

    dimension = initial_nodes.shape[1]
    corners_scale = mean_scale * corners_scale_proportion

    mean = get_mean(dimension=dimension, scale=mean_scale)

    corner_vectors = generate_corner_vectors(dimension=dimension, scale=corners_scale)
    corner_interpolation = interpolate_3d_corner_vectors(
        nodes=initial_nodes, base=base, corner_vectors=corner_vectors
    )
    return mean + corner_interpolation


# TODO: write in Numba
def get_top_indices(array, indices_count):
    unsorted_indices = np.argpartition(array, indices_count)[:indices_count]
    result = unsorted_indices[array[unsorted_indices].argsort()]
    # assert np.all(result == array.argsort()[:indices_count])
    return result


def approximate_internal(base_values, closest_nodes, closest_weights):
    return (base_values[closest_nodes] * closest_weights.reshape(*closest_weights.shape, 1)).sum(
        axis=1
    )


# @numba.njit
def get_interlayer_data_numba(
    base_nodes: np.ndarray,
    base_elements: np.ndarray,
    interpolated_nodes: np.ndarray,
    with_weights: bool,
    closest_count: int,
):
    _ = base_elements
    closest_distances = np.zeros((len(interpolated_nodes), closest_count))
    closest_nodes = np.zeros_like(closest_distances, dtype=np.int64)
    closest_weights = np.zeros_like(closest_distances) if with_weights else None

    # if with_weights:
    #     base_element_nodes = base_nodes[base_elements]
    #     base_nodes_min = base_element_nodes.min(axis=1)
    #     base_nodes_max = base_element_nodes.max(axis=1)
    for index, node in enumerate(interpolated_nodes):
        distances = nph.euclidean_norm_numba(base_nodes - node)
        closest_node_list = get_top_indices(distances, closest_count)
        if closest_weights is not None:
            closest_node_list = get_top_indices(distances, closest_count)
            selected_base_nodes = base_nodes[closest_node_list]

            if np.all(selected_base_nodes[0] == node):
                closest_weights[index, 0] = 1
            else:
                # Moore-Penrose pseudo-inverse
                # weights_internal = np.ascontiguousarray(node) @
                #  np.linalg.pinv(selected_base_nodes)
                # if np.min(weights_internal) > 0 and np.abs(np.sum(weights_internal) - 1) < 0.003:
                #     unnormalized_weights = weights_internal
                # else:
                unnormalized_weights = 1.0 / (distances[closest_node_list] ** 2)
                node_weights = unnormalized_weights / np.sum(unnormalized_weights)
                closest_weights[index, :] = node_weights

        closest_distance_list = distances[closest_node_list]
        closest_nodes[index, :] = closest_node_list
        closest_distances[index, :] = closest_distance_list

    return closest_nodes, closest_distances, closest_weights


# pylint: disable=invalid-name
# pylint: disable=too-many-arguments
@numba.njit(fastmath=True, error_model="numpy")  # parallel=True)  # parallel=True)#, fastmath=True)
def find_closest_nodes_numba(
    closest_nodes,
    closest_weights,
    interpolated_nodes,
    base_elements,
    element_centers,
    element_ball_radiuses,
    element_nodes_matrices_T,
    normalizing_element_nodes_T,
    ready_nodes_mask,
    nodes_query,
    cell_starts,
    node_cell,
):
    spacing = 0.05  # 0.03  # element_ball_radiuses.mean()

    cell_starts, node_cell, spacing = initialize_hasher_numba(
        interpolated_nodes,
        spacing=spacing,
        cell_starts=cell_starts,
        node_cell=node_cell,
    )
    ready_nodes_mask[:] = False
    node_weights = np.empty(4)
    # max_m = 0
    for element_id, element in enumerate(base_elements):
        # close_nodes_mask = ((element_centers[element_id] - interpolated_nodes) ** 2).sum(
        #     axis=1
        # ) <= element_ball_radiuses[element_id] ** 2

        query_size = query_hasher_numba(
            nodes_query=nodes_query,
            ready_nodes_mask=ready_nodes_mask,
            query_node=element_centers[element_id],
            max_dist=element_ball_radiuses[element_id],
            cell_starts=cell_starts,
            node_cell=node_cell,
            spacing=spacing,
        )

        # m = close_nodes_mask.sum() / len(close_nodes_mask)
        # if m > max_m:
        #     max_m = m

        # node_ids = np.argwhere(ready_nodes_mask | ~close_nodes_mask )
        for i in range(query_size):
            node_id = nodes_query[i]
            # if ready_nodes_mask[node_id]:
            #     continue
            node = interpolated_nodes[node_id]
            # TODO: check why loop is slower than masking and why masking doesn't work with parallel
            # if(((element_center - node) ** 2).sum() > element_ball_radius_squared):

            node_weights[:3] = np.linalg.solve(
                element_nodes_matrices_T[element_id], node - normalizing_element_nodes_T[element_id]
            )
            node_weights[3] = 1 - node_weights[:3].sum()  # weights sum to one

            # looking for weights that are closest to positive
            smallest_weight = np.min(node_weights)
            if smallest_weight > np.min(closest_weights[node_id, :]):
                closest_weights[node_id, :] = node_weights
                closest_nodes[node_id, :] = element
            if smallest_weight >= 0:
                # positive weights found, only one element can contain node
                ready_nodes_mask[node_id] = True


def get_interlayer_data_skinning_numba(
    base_nodes: np.ndarray,
    base_elements: np.ndarray,
    interpolated_nodes: np.ndarray,
    # padding: float,
):  # TODO: Use multithreading
    padding = 0.01
    dim = base_nodes.shape[1]
    nodes_count = len(interpolated_nodes)
    closest_count = dim + 1
    element_nodes_count = base_elements.shape[1]
    element_nodes = base_nodes[base_elements]

    closest_distances = np.zeros((nodes_count, closest_count))
    closest_nodes = np.full_like(closest_distances, fill_value=-1, dtype=np.int64)
    closest_weights = np.full_like(closest_distances, fill_value=-1e8)

    element_centers = element_nodes.mean(axis=1)
    element_center_distances = element_nodes - element_centers.reshape(-1, 1, dim).repeat(
        element_nodes_count, 1
    )
    element_ball_radiuses = np.linalg.norm(element_center_distances, axis=2).max(axis=1) + padding

    element_nodes_matrices_T = (element_nodes[:, :dim] - element_nodes[:, [dim]]).transpose(0, 2, 1)
    normalizing_element_nodes_T = element_nodes[:, [dim]].reshape(-1, 3)

    ready_nodes_mask = np.zeros(nodes_count, dtype=bool)
    nodes_query = np.zeros(nodes_count, dtype=np.int64)

    table_size = 2 * nodes_count
    cell_starts = np.zeros(table_size + 1, dtype=np.int64)
    node_cell = np.zeros(nodes_count, dtype=np.int64)

    np.save("./output/base_nodes", base_nodes)
    np.save("./output/base_elements", base_elements)
    np.save("./output/interpolated_nodes", interpolated_nodes)

    timer = Timer()
    with timer["find_closest_nodes_numba"]:
        m = find_closest_nodes_numba(
            closest_nodes=closest_nodes,
            closest_weights=closest_weights,
            interpolated_nodes=interpolated_nodes,
            base_elements=base_elements,
            element_centers=element_centers,
            element_ball_radiuses=element_ball_radiuses,
            element_nodes_matrices_T=element_nodes_matrices_T,
            normalizing_element_nodes_T=normalizing_element_nodes_T,
            ready_nodes_mask=ready_nodes_mask,
            nodes_query=nodes_query,
            cell_starts=cell_starts,
            node_cell=node_cell,
        )
    print(timer.to_dataframe())
    print(m)

    assert np.all(closest_nodes >= 0)  # For each node at least one element found
    print("Min weight", closest_weights.min())
    return closest_nodes, closest_distances, closest_weights


# pylint: disable=import-outside-toplevel
# pylint: disable=no-name-in-module
def get_interlayer_data_skinning_cython(
    base_nodes: np.ndarray,
    base_elements: np.ndarray,
    interpolated_nodes: np.ndarray,
    # padding: float,
):
    from cython_modules import weights

    # padding = 0.0
    element_radius_padding = 0.01

    int_type = np.int64
    # int_type= np.int32
    # float_type = np.float32
    float_type = np.float64

    dim = base_nodes.shape[1]
    nodes_count = len(interpolated_nodes)
    elements_count = len(base_elements)
    closest_count = dim + 1
    # element_nodes_count = base_elements.shape[1]

    spacing = 0.05
    table_size = 2 * elements_count

    table_size_proportion = 2
    table_size = table_size_proportion * nodes_count

    cell_starts = np.zeros(table_size + 1, dtype=int_type)
    node_cell = np.zeros(nodes_count, dtype=int_type)

    closest_distances = np.zeros((nodes_count, closest_count))
    closest_nodes = np.full_like(closest_distances, fill_value=-1, dtype=int_type)
    closest_weights = np.full_like(closest_distances, fill_value=-1e8, dtype=float_type)

    ready_nodes_mask = np.zeros(nodes_count, dtype=bool)
    nodes_query = np.zeros(nodes_count, dtype=int_type)
    # element_nodes = np.zeros((elements_count, 4, 3))

    assert dim == 3  # Not implemeted for 2D
    # t = time()
    weights.find_closest_nodes_cython(
        closest_nodes=closest_nodes,
        closest_weights=closest_weights,
        interpolated_nodes=interpolated_nodes.astype(float_type),
        base_elements=base_elements.astype(int_type),
        base_nodes=base_nodes,
        query_nodes=nodes_query,
        ready_nodes_mask=ready_nodes_mask,
        cell_starts=cell_starts,
        node_cell=node_cell,
        spacing=spacing,
        element_radius_padding=element_radius_padding,
    )
    # stop = time() - t

    # assert np.all(closest_nodes >= 0)  # For each node at least one element found
    # print("Min weight", closest_weights.min())
    # print("Time all ms: ", 1000 * stop)
    # print("Ready ", ready_nodes_mask.sum())
    # print("nodes ", nodes_count)
    return closest_nodes, closest_distances, closest_weights


def interpolate_nodes(
    base_nodes: np.ndarray,
    base_elements: np.ndarray,
    query_nodes: np.ndarray,
    # padding: float,
):
    # return get_interlayer_data_skinning_numba(
    #     base_nodes=base_nodes,
    #     base_elements=base_elements,
    #     interpolated_nodes=interpolated_nodes,
    #     # padding: float,
    # )

    return get_interlayer_data_skinning_cython(
        base_nodes=base_nodes,
        base_elements=base_elements,
        interpolated_nodes=query_nodes,
        # padding: float,
    )
