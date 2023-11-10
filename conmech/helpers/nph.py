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


def unstack_and_sum_columns(
    data: np.ndarray, dim: int, keepdims: bool = False
) -> np.ndarray:
    return np.sum(unstack(data, dim), axis=1, keepdims=keepdims)


def elementwise_dot(
    matrix_1: np.ndarray, matrix_2: np.ndarray, keepdims: bool = False
) -> np.ndarray:
    return (matrix_1 * matrix_2).sum(axis=1, keepdims=keepdims)


def get_occurances(data: np.ndarray) -> np.ndarray:
    return np.array(list(set(data.flatten())))


def close_modulo(value: np.ndarray, divider: Optional[int]) -> bool:
    if divider is None:
        return True
    return np.allclose(value % divider, 0.0) or np.allclose(value % divider, divider)


def euclidean_norm(vector: np.ndarray, keepdims=False) -> np.ndarray:
    data = (vector**2).sum(axis=-1, keepdims=keepdims)
    if isinstance(vector, np.ndarray):
        return np.sqrt(data)
    return data.sqrt()
    # return np.linalg.norm(vector, axis=-1)
    # return np.sqrt(np.sum(vector ** 2, axis=-1))[..., np.newaxis]


@numba.njit
def euclidean_norm_numba(vector: np.ndarray) -> np.ndarray:
    data = (vector**2).sum(axis=-1)
    return np.sqrt(data)


def get_normal(vector: np.ndarray, normal: np.ndarray) -> np.ndarray:
    return elementwise_dot(vector, normal, keepdims=True)


def get_normal_tangential(
    vector: np.ndarray, normal: np.ndarray
) -> Tuple[np.ndarray, np.ndarray]:
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
def get_node_index_numba(node, nodes):
    for i, n in enumerate(nodes):
        if np.sum(np.abs(node - n)) < 0.0001:
            return i
    raise ArgumentError


def generate_normal(rows: int, columns: int, scale: float) -> np.ndarray:
    return np.random.normal(loc=0.0, scale=scale * 0.5, size=[rows, columns])


@numba.njit(inline="always")
def length(p_1, p_2):
    return np.sqrt((p_1[0] - p_2[0]) ** 2 + (p_1[1] - p_2[1]) ** 2)


@numba.njit(inline="always")
def length_prb(edge, nodes):
    return np.sqrt(
        (nodes[edge[0]][0] - nodes[edge[1]][0]) ** 2
        + (nodes[edge[0]][1] - nodes[edge[1]][1]) ** 2
    )
