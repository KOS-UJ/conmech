import random

import numpy as np

from conmech.helpers import nph


def decide(scale):
    return np.random.uniform(low=0, high=1) < scale


def choose(options):
    return random.choice(options)


def get_corner_vectors_rotate(dim, scale):
    # 1 2
    # 0 3
    corner_vector = nph.generate_normal_circle(rows=1, columns=dim, scale=scale)
    corner_vectors = corner_vector * [[1, 1], [-1, 1], [-1, -1], [1, -1]]
    return corner_vectors


def get_corner_vectors_all(dimension, scale):
    corner_vectors = nph.generate_normal_circle(rows=dimension * 2, columns=dimension, scale=scale)
    return corner_vectors


def get_mean(scale, initial_nodes):
    return np.random.uniform(
        low=-scale,
        high=scale,
        size=(1, initial_nodes.shape[1]),
    )


def scale_nodes_to_square(nodes):
    scaled_nodes = (nodes - np.min(nodes, axis=0)) / (np.max(nodes, axis=0) - np.min(nodes, axis=0))
    return scaled_nodes


def interpolate_nodes(scaled_nodes, corner_vectors):
    dimension = scaled_nodes.shape[1]
    values = np.zeros_like(scaled_nodes)
    for i in range(dimension):
        coordinate_i = scaled_nodes[:, [i]]
        values += (
            coordinate_i * corner_vectors[i] + (1 - coordinate_i) * corner_vectors[i + dimension]
        )
    return values


def interpolate_four(
    initial_nodes: np.ndarray,
    scale: float,
    corners_scale_proportion: float,
    base: np.ndarray,
    interpolate_rotate: bool,
):
    dimension = initial_nodes.shape[1]
    if dimension != 2:
        raise NotImplementedError
    mean = get_mean(scale, initial_nodes)
    corners_scale = scale * corners_scale_proportion

    get_corner_vectors = get_corner_vectors_all
    # (
    #    get_corner_vectors_rotate if interpolate_rotate else get_corner_vectors_four
    # )

    corner_vectors = get_corner_vectors(dimension=dimension, scale=corners_scale)

    # orthonormal matrix; inverse equals transposition
    initial_denormalized_nodes = initial_nodes  # nph.get_in_base(initial_nodes, base.T)
    denormalized_scaled_nodes = scale_nodes_to_square(initial_denormalized_nodes)

    interpolated_denormalized_nodes = interpolate_nodes(
        scaled_nodes=denormalized_scaled_nodes,
        corner_vectors=corner_vectors,
    )

    interpolated_nodes = (
        interpolated_denormalized_nodes  # nph.get_in_base(interpolated_denormalized_nodes, base)
    )
    return mean + interpolated_nodes
