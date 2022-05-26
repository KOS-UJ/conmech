import random
from ctypes import ArgumentError

import numba
import numpy as np
from scipy.interpolate import interpn

from conmech.helpers import nph


def decide(scale):
    return np.random.uniform(low=0, high=1) < scale


def choose(options):
    return random.choice(options)


def get_mean(dimension, scale):
    return nph.generate_normal(rows=1, columns=dimension, sigma=scale / 3)


# def get_corner_vectors_rotate(dimension, scale):
#     if dimension != 2:
#         raise NotImplementedError
#     # 1 2
#     # 0 3
#     corner_vector = nph.generate_normal(rows=1, columns=dimension, sigma=scale / 3)
#     corner_vectors = corner_vector * [[1, 1], [-1, 1], [-1, -1], [1, -1]]
#     return corner_vectors


def generate_mesh_corner_scalars(dimension: int, scale: float):
    random_vector = nph.generate_normal(rows=(2**dimension), columns=1, sigma=scale / 3)
    clipped_vector = np.maximum(-scale, np.minimum(random_vector, scale))
    normalized_cliped_vector = clipped_vector - np.mean(clipped_vector)
    return 1 + normalized_cliped_vector


def generate_corner_vectors(dimension: int, scale: float):
    corner_vectors = nph.generate_normal(rows=(2**dimension), columns=dimension, sigma=scale / 3)
    normalized_corner_vectors = corner_vectors - np.mean(corner_vectors, axis=0)
    return normalized_corner_vectors


def scale_nodes_to_cube(nodes):
    nodes_min = np.min(nodes, axis=0)
    nodes_max = np.max(nodes, axis=0)
    scaled_nodes = (nodes - nodes_min) / (nodes_max - nodes_min)
    return scaled_nodes


def interpolate_scaled_nodes(scaled_nodes: np.ndarray, corner_vectors: np.ndarray):
    if np.min(scaled_nodes) < 0 or np.max(scaled_nodes) > 1:
        raise ArgumentError

    dimension = scaled_nodes.shape[-1]
    segment = np.linspace(0, 1, 2)
    grid = (segment,) * dimension

    def reshape_corner_data(data: np.ndarray):
        if dimension == 2:
            return data.reshape(2, 2, -1)
        elif dimension == 3:
            return data.reshape(2, 2, 2, -1)
        else:
            raise ArgumentError

    values = reshape_corner_data(corner_vectors)
    # interpolating_function = RegularGridInterpolator(points=corner_nodes, values=corner_vectors)
    result = interpn(points=grid, values=values, xi=scaled_nodes, method="linear")
    return result


# def interpolate_scaled_nodes_new(scaled_nodes: np.ndarray, corner_vectors: np.ndarray):
#     if np.min(scaled_nodes) < 0 or np.max(scaled_nodes) > 1:
#         raise ArgumentError
#     corner_nodes = np.array([[1.0, 1.0], [0.0, 1.0], [1.0, 0.0], [0.0, 0.0]])
#     closest_nodes, closest_weights, closest_distances = get_interlayer_data(
#         base_nodes=corner_nodes, interpolated_nodes=scaled_nodes, closest_count=len(corner_nodes)
#     )
#     result = approximate_internal(
#         base_values=corner_vectors, closest_nodes=closest_nodes, closest_weights=closest_weights
#     )
#     return result


def interpolate_scaled_nodes_old(scaled_nodes: np.ndarray, corner_vectors: np.ndarray):
    if np.min(scaled_nodes) < 0 or np.max(scaled_nodes) > 1:
        raise ArgumentError
    input_dim = scaled_nodes.shape[-1]
    output_dim = corner_vectors.shape[-1]
    values = np.zeros((scaled_nodes.shape[0], output_dim))
    for i in range(input_dim):
        coordinate_i = scaled_nodes[..., [i]]
        values += (
            coordinate_i * corner_vectors[i] + (1 - coordinate_i) * corner_vectors[i + input_dim]
        ) / input_dim
    return values


def interpolate_corner_vectors(nodes: np.ndarray, base: np.ndarray, corner_vectors: np.ndarray):
    # orthonormal matrix; inverse equals transposition
    upward_nodes = nph.get_in_base(nodes, base.T)
    scaled_nodes = scale_nodes_to_cube(upward_nodes)
    upward_vectors_interpolation = interpolate_scaled_nodes(
        scaled_nodes=scaled_nodes,
        corner_vectors=corner_vectors,
    )
    assert np.allclose(np.min(upward_vectors_interpolation, axis=0), np.min(corner_vectors, axis=0))
    assert np.allclose(np.max(upward_vectors_interpolation, axis=0), np.max(corner_vectors, axis=0))
    assert np.allclose(np.mean(upward_vectors_interpolation, axis=0), [0, 0], atol=0.01)

    vectors_interpolation = nph.get_in_base(upward_vectors_interpolation, base)
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
def get_interlayer_data(base_nodes: np.ndarray, interpolated_nodes: np.ndarray, closest_count: int):
    closest_distances = np.zeros((len(interpolated_nodes), closest_count))
    closest_nodes = np.zeros_like(closest_distances, dtype=np.int64)
    closest_weights = np.zeros_like(closest_distances)

    for index, node in enumerate(interpolated_nodes):
        distances = nph.euclidean_norm_numba(base_nodes - node)
        closest_node_list = distances.argsort()[:closest_count]
        closest_distance_list = distances[closest_node_list]
        selected_base_nodes = base_nodes[closest_node_list]

        # if closest_distance_list[0] < 1.0e-8:
        #     closest_weight_list = np.zeros_like(closest_distance_list)
        #     closest_weight_list[0] = 1
        # else:
        # Moore-Penrose pseudo-inverse
        closest_weight_list = np.ascontiguousarray(node) @ np.linalg.pinv(selected_base_nodes)
        assert np.min(closest_weight_list) >= 0 and np.sum(closest_weight_list) == 1

        closest_nodes[index, :] = closest_node_list
        closest_weights[index, :] = closest_weight_list
        closest_distances[index, :] = closest_distance_list

    return closest_nodes, closest_weights, closest_distances


def approximate_internal(base_values, closest_nodes, closest_weights):
    return (base_values[closest_nodes] * closest_weights.reshape(*closest_weights.shape, 1)).sum(
        axis=1
    )
