import random
from ctypes import ArgumentError

import jax
import jax.numpy as jnp
import numba
import numpy as np
from tqdm import tqdm

from conmech.helpers import jxh, lnh, nph
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
                weights = unnormalized_weights / np.sum(unnormalized_weights)
                closest_weights[index, :] = weights

        closest_distance_list = distances[closest_node_list]
        closest_nodes[index, :] = closest_node_list
        closest_distances[index, :] = closest_distance_list

    return closest_nodes, closest_distances, closest_weights

@numba.njit()  # parallel=True)  # parallel=True)#, fastmath=True)
def iterate(
    closest_nodes,
    closest_weights,
    interpolated_nodes,
    base_elements,
    element_centers,
    element_ball_radiuses_squared,
    P_T,
    p4,
):
    N = len(interpolated_nodes)
    T = len(p4)
    for i in numba.prange(N):
        node = interpolated_nodes[i]
        closest_index_mask = -1
        closest_smallest_weight = -1e8
        weights = np.empty(4)

        # TODO: check why loop is slower than masking and why masking doesn't work with parallel
        outside_mask = ((element_centers - node) ** 2).sum(axis=1) > element_ball_radiuses_squared
        for j in range(T):
            # if (
            #     ((element_centers[j] - node) ** 2).sum() > element_ball_radiuses_squared[j]
            # ):
            if outside_mask[j]:
                continue

            weights[:3] = np.linalg.solve(P_T[j], node - p4[j])
            weights[3] = 1 - weights[:3].sum()  # weights sum to one

            # looking for weights that are closest to boeing positive
            smallest_weight = np.min(weights)
            if smallest_weight > closest_smallest_weight:
                closest_smallest_weight = smallest_weight
                closest_index_mask = j
                closest_weights[i, :] = weights
            if smallest_weight >= 0:  # positive weights found, only one element can contain node
                break

        closest_nodes[i, :] = base_elements[closest_index_mask]


def get_interlayer_data_NEW(
    base_nodes: np.ndarray,
    base_elements: np.ndarray,
    interpolated_nodes: np.ndarray,
    # padding: float,
):
    padding = 0.1
    closest_count = 4
    closest_distances = np.zeros((len(interpolated_nodes), closest_count))
    closest_nodes = np.empty_like(closest_distances, dtype=np.int64)
    closest_weights = np.empty_like(closest_distances)
    closest_nodes[:] = -1
    closest_weights[:] = -1e8

    weights = np.zeros((len(base_elements), closest_count))

    dim = base_nodes.shape[1]
    element_nodes_count = base_elements.shape[1]

    element_nodes = base_nodes[base_elements]

    element_centers = np.array(element_nodes.mean(axis=1))
    element_center_distances = element_nodes - element_centers.reshape(-1, 1, dim).repeat(
        element_nodes_count, 1
    )
    element_ball_radiuses = np.linalg.norm(element_center_distances, axis=2).max(axis=1) + padding
    mask = np.zeros_like(element_ball_radiuses, dtype=np.bool)

    p4 = element_nodes[:, [3]]
    P = element_nodes[:, :3] - p4
    P_T = P.transpose(0, 2, 1)

    timer = Timer()
    with timer["iterate"]:
        iterate(
            closest_nodes,
            closest_weights,
            interpolated_nodes,
            base_elements,
            element_centers,
            element_ball_radiuses**2,
            P_T,
            p4.reshape(-1, 3),
        )
    assert np.all(closest_nodes >= 0)  # For each node at least one element found
    print("Min weight", closest_weights.min())

    a = 0

    # for i, element in enumerate(tqdm(base_elements)):
    #     mask = (
    #         np.linalg.norm(element_centers[i] - interpolated_nodes, axis=1)
    #         - element_ball_radiuses[i]
    #         <= 0
    #     )
    #     b = (interpolated_nodes - p4[i])[mask]

    #     x = np.linalg.solve(P_T[i], b.T)

    with timer["numpy"]:
        for index, node in enumerate(tqdm(interpolated_nodes)):
            # mask = np.asarray(mask_fun(node))
            mask = np.linalg.norm(element_centers - node, axis=1) - element_ball_radiuses <= 0

            b = (node - p4[mask]).transpose(0, 2, 1)
            weights[mask, :3] = np.linalg.solve(P_T[mask], b).reshape(-1, 3)
            weights[mask, 3] = 1 - weights[mask, :3].sum(1)

            masked_weights = weights[mask]
            closest_index_mask = np.argmin((-masked_weights).max(axis=1))
            closest_nodes[index, :] = base_elements[mask][closest_index_mask]
            closest_weights[index, :] = masked_weights[closest_index_mask]

    print(timer.to_dataframe())
    return closest_nodes, closest_distances, closest_weights


def approximate_internal(base_values, closest_nodes, closest_weights):
    return (base_values[closest_nodes] * closest_weights.reshape(*closest_weights.shape, 1)).sum(
        axis=1
    )
