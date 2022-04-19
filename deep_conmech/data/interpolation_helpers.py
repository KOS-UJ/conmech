import random

import numba
import numpy as np

from conmech.helpers import nph
from conmech.properties.mesh_properties import MeshProperties


@numba.njit
def weighted_mean_numba(v1, v2, scale):
    return v1 * (1 - scale) + v2 * scale


@numba.njit
def interpolate_point_numba(initial_point, corner_vectors, scale_x, scale_y):
    min = [0.0, 0.0]  # TODO #65
    x_scale = (initial_point[0] - min[0]) / scale_x
    y_scale = (initial_point[1] - min[1]) / scale_y

    top_scaled = weighted_mean_numba(corner_vectors[1], corner_vectors[2], x_scale)
    bottom_scaled = weighted_mean_numba(corner_vectors[0], corner_vectors[3], x_scale)

    scaled = weighted_mean_numba(bottom_scaled, top_scaled, y_scale)
    return scaled


@numba.njit
def interpolate_numba(initial_nodes, corner_vectors, scale_x, scale_y):
    nodes = np.zeros_like(initial_nodes)
    for i in range(initial_nodes.shape[0]):
        nodes[i] = interpolate_point_numba(initial_nodes[i], corner_vectors, scale_x, scale_y)
    return nodes


def decide(scale):
    return np.random.uniform(low=0, high=1) < scale


def choose(options):
    return random.choice(options)


def get_corner_vectors_rotate(dim, scale):
    # 1 2
    # 0 3
    corner_vector = nph.draw_normal_circle(rows=1, columns=dim, scale=scale)
    corner_vectors = corner_vector * [[1, 1], [-1, 1], [-1, -1], [1, -1]]
    return corner_vectors


def get_corner_vectors_four(dim, scale):
    corner_vectors = nph.draw_normal_circle(rows=4, columns=dim, scale=scale)
    return corner_vectors


def get_mean(scale, initial_nodes):
    return np.random.uniform(
        low=-scale,
        high=scale,
        size=(1, initial_nodes.shape[1]),
    )


def interpolate_four(
    initial_nodes: np.ndarray,
    scale: float,
    corners_scale_proportion: float,
    mesh_prop: MeshProperties,
    base: np.ndarray,
    interpolate_rotate: bool,
):
    mean = get_mean(scale, initial_nodes)
    corners_scale = scale * corners_scale_proportion

    get_corner_vectors = (
        get_corner_vectors_rotate if interpolate_rotate else get_corner_vectors_four
    )

    # orthonormal matrix; inverse equals transposition
    initial_denormalized_nodes = nph.get_in_base(initial_nodes, base.T)
    interpolated_denormalized_nodes = interpolate_numba(
        initial_nodes=initial_denormalized_nodes,
        corner_vectors=get_corner_vectors(dim=initial_nodes.shape[1], scale=corners_scale),
        scale_x=mesh_prop.scale_x,
        scale_y=mesh_prop.scale_y,
    )
    interpolated_nodes = nph.get_in_base(interpolated_denormalized_nodes, base)
    return mean + interpolated_nodes
