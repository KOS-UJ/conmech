import random
from ctypes import ArgumentError

import numba
import numpy as np

from conmech.helpers import lnh, nph


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

        def interpolate(dim, tag, x, y, z, lc):
            interpolated_values_1 = interpolate_node_numba(corner_values, x)
            reinterpolated_values_2 = interpolate_node_numba(interpolated_values_1, y)
            return reinterpolated_values_2
            # values_y_0 = corner_vectors[0] * x + corner_vectors[1] * (1 - x)
            # values_y_1 = corner_vectors[2] * x + corner_vectors[3] * (1 - x)
            # return values_y_0 * y + values_y_1 * (1 - y)

        return interpolate

    if len(corner_vectors) == 8:
        corner_values = corner_vectors.reshape(2, 2, 2, -1)

        def interpolate(dim, tag, x, y, z, lc):
            interpolated_values_1 = interpolate_node_numba(corner_values, x)
            interpolated_values_2 = interpolate_node_numba(interpolated_values_1, y)
            interpolated_values_3 = interpolate_node_numba(interpolated_values_2, z)
            return interpolated_values_3
            # values_y_0 = corner_vectors[0] * x + corner_vectors[1] * (1 - x)
            # values_y_1 = corner_vectors[2] * x + corner_vectors[3] * (1 - x)
            # values_y_2 = corner_vectors[4] * x + corner_vectors[5] * (1 - x)
            # values_y_3 = corner_vectors[6] * x + corner_vectors[7] * (1 - x)
            # values_z_0 = values_y_0 * y + values_y_1 * (1 - y)
            # values_z_1 = values_y_2 * y + values_y_3 * (1 - y)
            # return values_z_0 * z + values_z_1 * (1 - z)

        return interpolate
    else:
        raise ArgumentError


def scale_nodes_to_cube(nodes):
    nodes_min = np.min(nodes, axis=0)
    nodes_max = np.max(nodes, axis=0)
    scaled_nodes = (nodes - nodes_min) / (nodes_max - nodes_min)
    return scaled_nodes


def interpolate_corner_vectors(nodes: np.ndarray, base: np.ndarray, corner_vectors: np.ndarray):
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
    corner_interpolation = interpolate_corner_vectors(
        nodes=initial_nodes, base=base, corner_vectors=corner_vectors
    )
    return mean + corner_interpolation


@numba.njit
def get_interlayer_data(
    base_nodes: np.ndarray, interpolated_nodes: np.ndarray, closest_count: int, with_weights: bool
):
    closest_distances = np.zeros((len(interpolated_nodes), closest_count))
    closest_nodes = np.zeros_like(closest_distances, dtype=np.int64)
    closest_weights = np.zeros_like(closest_distances) if with_weights else None

    for index, node in enumerate(interpolated_nodes):
        distances = nph.euclidean_norm_numba(base_nodes - node)
        closest_node_list = distances.argsort()[:closest_count]
        closest_distance_list = distances[closest_node_list]
        selected_base_nodes = base_nodes[closest_node_list]
        if with_weights:
            # Moore-Penrose pseudo-inverse
            closest_weight_list = np.ascontiguousarray(node) @ np.linalg.pinv(selected_base_nodes)
            # assert np.min(closest_weight_list) >= 0
            # assert np.sum(closest_weight_list) == 1
            closest_weights[index, :] = closest_weight_list

        closest_nodes[index, :] = closest_node_list
        closest_distances[index, :] = closest_distance_list

    return closest_nodes, closest_distances, closest_weights


def approximate_internal(base_values, closest_nodes, closest_weights):
    return (base_values[closest_nodes] * closest_weights.reshape(*closest_weights.shape, 1)).sum(
        axis=1
    )
