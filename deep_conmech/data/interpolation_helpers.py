import random
from ctypes import ArgumentError

import numba
import numpy as np
import scipy
from tqdm import tqdm

from conmech.helpers import lnh, nph
from deep_conmech.training_config import CLOSEST_BOUNDARY_COUNT, CLOSEST_COUNT


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

        return interpolate

    if len(corner_vectors) == 8:
        corner_values = corner_vectors.reshape(2, 2, 2, -1)

        def interpolate(dim, tag, x, y, z, lc):
            interpolated_values_1 = interpolate_node_numba(corner_values, x)
            interpolated_values_2 = interpolate_node_numba(interpolated_values_1, y)
            interpolated_values_3 = interpolate_node_numba(interpolated_values_2, z)
            return interpolated_values_3

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
def same_side_numba(v1, v2, v3, v4, p):
    epsilon = 1e-10
    normal = np.cross(v2 - v1, v3 - v1)
    normal = normal / np.linalg.norm(normal)
    dotV4 = np.dot(normal, v4 - v1)
    dotP = np.dot(normal, p - v1)
    return np.abs(dotP) < epsilon or np.sign(dotV4) == np.sign(dotP)


@numba.njit
def point_in_tetrahedron_numba(nodes, p):
    for n in nodes:
        if np.all(p == n):
            return True
    v1, v2, v3, v4 = nodes
    return (
        same_side_numba(v1, v2, v3, v4, p)
        and same_side_numba(v2, v3, v4, v1, p)
        and same_side_numba(v3, v4, v1, v2, p)
        and same_side_numba(v4, v1, v2, v3, p)
    )


# @numba.njit
def reformat_sparse_numba(
    dense_nodes: np.ndarray,
    dense_elements: np.ndarray,
    dense_boundary_centers: np.ndarray,
    sparse_nodes: np.ndarray,
):
    sparse_nodes_copy = sparse_nodes.copy()
    nodes_out = 0

    dense_nodes_min = dense_nodes[dense_elements].min(axis=1)
    dense_nodes_max = dense_nodes[dense_elements].max(axis=1)

    for index, node in enumerate(sparse_nodes):
        if index % 100 == 0:
            print(index)

        lower_bound = np.all(dense_nodes_min <= node, axis=1)
        upper_bound = np.all(node <= dense_nodes_max, axis=1)
        bounding_elements_indices = np.where(np.logical_and(lower_bound, upper_bound))[0]

        closest_element_index = -1
        for i in bounding_elements_indices:
            if point_in_tetrahedron_numba(dense_nodes[dense_elements[i]], node):
                closest_element_index = i
                break

        if closest_element_index < 0:
            distances = nph.euclidean_norm_numba(dense_nodes - node)
            closest_node_list = distances.argsort()[:4]
            selected_base_nodes = dense_nodes[closest_node_list]

            closest_weight_list = np.ascontiguousarray(node) @ np.linalg.pinv(selected_base_nodes)

            d1 = np.argmin(distances)
            closest_node = dense_nodes[d1]
            # distances2 = nph.euclidean_norm_numba(dense_boundary_centers - node)
            # d2 = np.argmin(distances2)
            # # if d1<d2:
            # #     closest_node = dense_nodes[d1]
            # # else:
            # closest_node = dense_boundary_centers[d2]
            sparse_nodes[index] = closest_node
            nodes_out += 1

    print("Testing repetitions")
    for index1, node1 in enumerate(sparse_nodes):
        for index2, node2 in enumerate(sparse_nodes):
            if index1 != index2 and np.all(np.equal(node1, node2)):
                raise ArgumentError

    print("Done")
    return sparse_nodes, nodes_out


@numba.njit
def get_closest_element_index_numba(elements_indices, node, element_nodes):
    for i in elements_indices:
        if point_in_tetrahedron_numba(element_nodes[i], node):
            return i
    return None

# TODO: write in Numba
def get_top_indices(array, indices_count):
    unsorted_indices = np.argpartition(array, indices_count)[:indices_count]
    result = unsorted_indices[array[unsorted_indices].argsort()]
    #assert np.all(result == array.argsort()[:indices_count])
    return result


# @numba.njit
def get_interlayer_data_numba(
    base_nodes: np.ndarray,
    base_elements: np.ndarray,
    interpolated_nodes: np.ndarray,
    with_weights: bool,
    closest_count: int,
):
    # closest_count = CLOSEST_COUNT
    # assert closest_count == 4
    #nodes_out = 0
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
                unnormalized_weights = 1.0 / (distances[closest_node_list] ** 2)
                weights = unnormalized_weights / np.sum(unnormalized_weights)
                closest_weights[index, :] = weights

                # weights2 = np.ascontiguousarray(
                #     np.hstack((node, np.ones((1))))
                # ) @ np.linalg.inv(np.hstack((selected_base_nodes, np.ones((4, 1)))))

                # if np.min(weights2) >= 0. and np.sum(weights2) == 1.0:
                #     assert np.linalg.norm(node - weights2 @ selected_base_nodes) < np.linalg.norm(node - weights @ selected_base_nodes)


            # lower_bound = np.all(base_nodes_min <= node, axis=1)
            # upper_bound = np.all(node <= base_nodes_max, axis=1)
            # bounding_elements_indices = np.where(np.logical_and(lower_bound, upper_bound))[0]

            # closest_element_index = get_closest_element_index_numba(
            #     elements_indices=bounding_elements_indices,
            #     node=node,
            #     element_nodes=base_element_nodes
            # )

            # if closest_element_index is None:
            #     closest_node_list = get_top_indices(distances, closest_count)
            #     closest_weights[index, 0] = 1
            #     # raise ArgumentError
            #     nodes_out += 1
            # else:
            #     closest_element = base_elements[closest_element_index]
            #     closest_node_list = closest_element[distances[closest_element].argsort()]
            #     selected_base_nodes = base_nodes[closest_node_list]

            #     if np.all(selected_base_nodes[0] == node):
            #         closest_weights[index, 0] = 1
            #     else:
            #         closest_weights[index, :] = np.ascontiguousarray(
            #             np.hstack((node, np.ones((1))))
            #         ) @ np.linalg.inv(np.hstack((selected_base_nodes, np.ones((4, 1)))))

            #         assert np.min(closest_weights[index, :]) > -0.001
            #         assert np.abs(np.sum(closest_weights[index, :]) - 1.0) < 0.001
            #         assert np.linalg.norm(node - closest_weights[index, :] @ selected_base_nodes) < 1e-10

        closest_distance_list = distances[closest_node_list]
        closest_nodes[index, :] = closest_node_list
        closest_distances[index, :] = closest_distance_list

    # if closest_weights is not None:
    #     assert np.mean(closest_weights) == 0.25
    return closest_nodes, closest_distances, closest_weights


def approximate_internal(base_values, closest_nodes, closest_weights):
    return (base_values[closest_nodes] * closest_weights.reshape(*closest_weights.shape, 1)).sum(
        axis=1
    )
