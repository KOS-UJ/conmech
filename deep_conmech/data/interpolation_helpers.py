import random

import numpy as np

from conmech.helpers import nph


def decide(scale):
    return np.random.uniform(low=0, high=1) < scale


def choose(options):
    return random.choice(options)


def get_mean(dimension, scale):
    return np.random.uniform(
        low=-scale,
        high=scale,
        size=(1, dimension),
    )


def get_corner_vectors_rotate(dimension, scale):
    if dimension != 2:
        raise NotImplementedError
    # 1 2
    # 0 3
    corner_vector = nph.generate_normal(rows=1, columns=dimension, scale=scale)
    corner_vectors = corner_vector * [[1, 1], [-1, 1], [-1, -1], [1, -1]]
    return corner_vectors


def get_corner_vectors_all(dimension, scale):
    corner_vectors = nph.generate_normal(rows=dimension * 2, columns=dimension, scale=scale)
    return corner_vectors


def scale_nodes_to_square(nodes):
    nodes_min = np.min(nodes, axis=0)
    nodes_max = np.max(nodes, axis=0)
    scaled_nodes = (nodes - nodes_min) / (nodes_max - nodes_min)
    return scaled_nodes


def interpolate_nodes(scaled_nodes, corner_vectors):
    input_dim = scaled_nodes.shape[-1]
    output_dim = corner_vectors.shape[-1]
    values = np.zeros((scaled_nodes.shape[0], output_dim))
    for i in range(input_dim):
        coordinate_i = scaled_nodes[..., [i]]
        values += (
            coordinate_i * corner_vectors[i] + (1 - coordinate_i) * corner_vectors[i + input_dim]
        ) / input_dim
    return values


def get_nodes_interpolation(nodes: np.ndarray, base: np.ndarray, corner_vectors: np.ndarray):
    # orthonormal matrix; inverse equals transposition
    denormalized_nodes = nph.get_in_base(nodes, base.T)
    denormalized_scaled_nodes = denormalized_nodes  # scale_nodes_to_square(denormalized_nodes)

    denormalized_nodes_interpolation = interpolate_nodes(
        scaled_nodes=denormalized_scaled_nodes,
        corner_vectors=corner_vectors,
    )
    nodes_interpolation = (
        denormalized_nodes_interpolation  # nph.get_in_base(denormalized_nodes_interpolation, base)
    )
    return nodes_interpolation


def interpolate_four(
    nodes: np.ndarray,
    mean_scale: float,
    corners_scale_proportion: float,
    base: np.ndarray,
    interpolate_rotate: bool,
    zero_out_proportion: float = 0,
):
    if decide(zero_out_proportion):
        return np.zeros_like(nodes)

    dimension = nodes.shape[1]
    corners_scale = mean_scale * corners_scale_proportion

    mean = get_mean(dimension=dimension, scale=mean_scale)

    get_corner_vectors = get_corner_vectors_rotate if interpolate_rotate else get_corner_vectors_all
    corner_vectors = get_corner_vectors(dimension=dimension, scale=corners_scale)
    nodes_interpolation = get_nodes_interpolation(
        nodes=nodes, base=base, corner_vectors=corner_vectors
    )

    return mean + nodes_interpolation
