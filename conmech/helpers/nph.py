"""
numpy helpers
"""
from ctypes import ArgumentError
from typing import Optional, Tuple

import numba
import numpy as np


def stack(data: np.ndarray) -> np.ndarray:
    return data.T.flatten()


def stack_column(data: np.ndarray) -> np.ndarray:
    return data.T.flatten().reshape(-1, 1)


stack_column_numba = numba.njit(stack_column)


def unstack(vector: np.ndarray, dim: int) -> np.ndarray:
    return vector.reshape(-1, dim, order="F")


def unstack_and_sum_columns(data: np.ndarray, dim: int, keepdims: bool = False) -> np.ndarray:
    return np.sum(unstack(data, dim), axis=1, keepdims=keepdims)


def elementwise_dot(matrix_1: np.ndarray, matrix_2: np.ndarray, keepdims: bool = False) -> np.ndarray:
    return (matrix_1 * matrix_2).sum(axis=1, keepdims=keepdims)


def get_occurances(data: np.ndarray) -> np.ndarray:
    return np.array(list(set(data.flatten())))


def close_modulo(value: np.ndarray, divider: Optional[int]) -> bool:
    if divider is None:
        return True
    return np.allclose(value % divider, 0.0) or np.allclose(value % divider, divider)


def euclidean_norm(vector: np.ndarray, keepdims=False) -> np.ndarray:
    data = (vector ** 2).sum(axis=-1, keepdims=keepdims)
    if isinstance(vector, np.ndarray):
        return np.sqrt(data)
    return data.sqrt()
    # return np.linalg.norm(vector, axis=-1)
    # return np.sqrt(np.sum(vector ** 2, axis=-1))[..., np.newaxis]


@numba.njit
def euclidean_norm_numba(vector: np.ndarray) -> np.ndarray:
    data = (vector ** 2).sum(axis=-1)
    return np.sqrt(data)


@numba.njit
def normalize_euclidean_numba(data: np.ndarray) -> np.ndarray:
    norm = euclidean_norm_numba(data)
    reshaped_norm = norm if data.ndim == 1 else norm.reshape(-1, 1)
    return data / reshaped_norm


def get_normal(vector: np.ndarray, normal: np.ndarray) -> np.ndarray:
    return elementwise_dot(vector, normal, keepdims=True)


def get_normal_tangential(vector: np.ndarray, normal: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    normal_vector = get_normal(vector, normal)
    tangential_vector = vector - (normal_vector * normal)
    return normal_vector, tangential_vector


def get_tangential(vector: np.ndarray, normal: np.ndarray) -> np.ndarray:
    _, tangential_vector = get_normal_tangential(vector, normal)
    return tangential_vector


@numba.njit
def get_tangential_numba(vector: np.ndarray, normal: np.ndarray) -> np.ndarray:
    normal_vector = vector @ normal
    tangential_vector = vector - (normal_vector * normal)
    return tangential_vector


def get_tangential_2d(normal: np.ndarray) -> np.ndarray:
    return np.array((normal[..., 1], -normal[..., 0])).T


def complete_base(base_seed: np.ndarray, closest_seed_index: int = 0) -> np.ndarray:
    dim = base_seed.shape[-1]
    normalized_base_seed = normalize_euclidean_numba(base_seed)
    if dim == 2:
        unnormalized_base = orthogonalize_gram_schmidt(normalized_base_seed)
    elif dim == 3:
        rolled_base_seed = np.roll(normalized_base_seed, -closest_seed_index, axis=0)
        unnormalized_rolled_base = orthogonalize_gram_schmidt(rolled_base_seed)
        unnormalized_base = np.roll(unnormalized_rolled_base, closest_seed_index, axis=0)
    else:
        raise ArgumentError
    base = normalize_euclidean_numba(unnormalized_base)
    return base


def orthogonalize_gram_schmidt(vectors: np.ndarray) -> np.ndarray:
    # Gramm-schmidt orthog.
    b0 = vectors[0]
    if len(vectors) == 1:
        return np.array((b0))

    b1 = vectors[1] - (vectors[1] @ b0) * b0
    if len(vectors) == 2:
        return np.array((b0, b1))

    # MGS for stability
    w2 = vectors[2] - (vectors[2] @ b0) * b0
    b2 = w2 - (w2 @ b1) * b1
    # nx = np.cross(ny,nz)
    return np.array((b0, b1, b2))


def get_in_base(vectors: np.ndarray, base: np.ndarray) -> np.ndarray:
    return vectors @ base.T


@numba.njit
def len_x_numba(corners):
    return corners[2] - corners[0]


@numba.njit
def len_y_numba(corners):
    return corners[3] - corners[1]


@numba.njit
def min_numba(corners):
    return [corners[0], corners[1]]


@numba.njit
def max_numba(corners):
    return [corners[2], corners[3]]


@numba.njit
def get_node_index_numba(node, nodes):
    for i, n in enumerate(nodes):
        if np.sum(np.abs(node - n)) < 0.0001:
            return i
    raise ArgumentError


def generate_normal(rows: int, columns: int, scale: float) -> np.ndarray:
    return np.random.normal(loc=0.0, scale=scale * 0.5, size=[rows, columns])


def generate_uniform_circle(rows: int, columns: int, low: float, high: float) -> np.ndarray:
    result = generate_normal(rows=rows, columns=columns, scale=1.0)
    normalized_result = normalize_euclidean_numba(result)
    radius = np.random.uniform(low=low, high=high, size=[rows, 1])
    return radius * normalized_result


def append_euclidean_norm(data: np.ndarray) -> np.ndarray:
    return np.hstack((data, euclidean_norm(data, keepdims=True)))


@numba.njit(inline="always")
def length(p_1, p_2):
    return np.sqrt((p_1[0] - p_2[0]) ** 2 + (p_1[1] - p_2[1]) ** 2)

# @numba.njit
# def calculate_angle_numba(new_up_vector):
#     old_up_vector = np.array([0., 1.])
#     angle = (2 * (new_up_vector[0] >= 0) - 1) * np.arccos(np.dot(new_up_vector, old_up_vector))
#     return angle
#
# @numba.njit
# def rotate_numba(vectors, angle):
#     s = np.sin(angle)
#     c = np.cos(angle)
#
#     rotated_vectors = np.zeros_like(vectors)
#     rotated_vectors[:, 0] = vectors[:, 0] * c - vectors[:, 1] * s
#     rotated_vectors[:, 1] = vectors[:, 0] * s + vectors[:, 1] * c
#
#     return rotated_vectors
