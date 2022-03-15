import random

import numpy as np
from numba import njit
from conmech.helpers import nph


@njit
def weighted_mean_numba(v1, v2, scale):
    return v1 * (1 - scale) + v2 * scale


@njit
def interpolate_point_numba(initial_point, corner_vectors, scale_x, scale_y):
    min = [0.0, 0.0]  #########
    x_scale = (initial_point[0] - min[0]) / scale_x
    y_scale = (initial_point[1] - min[1]) / scale_y

    top_scaled = weighted_mean_numba(corner_vectors[1], corner_vectors[2], x_scale)
    bottom_scaled = weighted_mean_numba(corner_vectors[0], corner_vectors[3], x_scale)

    scaled = weighted_mean_numba(bottom_scaled, top_scaled, y_scale)
    return scaled


@njit
def interpolate_numba(count, initial_nodes, corner_vectors, scale_x, scale_y):
    dim = initial_nodes.shape[1]
    result = np.zeros((count, dim))
    for i in range(count):
        initial_point = initial_nodes[i]
        result[i] = interpolate_point_numba(
            initial_point, corner_vectors, scale_x, scale_y
        )
    return result


def decide(scale):
    return np.random.uniform(low=0, high=1) < scale


def choose(list):
    return random.choice(list)


def get_corner_vectors_rotate(scale):
    # 1 2
    # 0 3
    corner_vector = nph.get_random_normal_circle(1, scale)
    corner_vectors = corner_vector * [[1, 1], [-1, 1], [-1, -1], [1, -1]]
    return corner_vectors


def get_corner_vectors_four(scale):
    corner_vectors = nph.get_random_normal_circle(4, scale)
    return corner_vectors


def interpolate_rotate(count, initial_nodes, randomization_scale, setting_scale):
    return interpolate_numba(
        count,
        initial_nodes,
        get_corner_vectors_rotate(randomization_scale),
        setting_scale,
    )


def interpolate_four(count, initial_nodes, randomization_scale, setting_scale):
    return interpolate_numba(
        count,
        initial_nodes,
        get_corner_vectors_four(randomization_scale),
        setting_scale,
    )
