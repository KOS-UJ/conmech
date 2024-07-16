"""
numpy helpers
"""

from ctypes import ArgumentError
from typing import Optional, Tuple

import numba
import numpy as np


def stack_column(data: np.ndarray) -> np.ndarray:
    return data.T.flatten().reshape(-1, 1)


stack_column_numba = numba.njit(stack_column)


def unstack(vector: np.ndarray, dim: int) -> np.ndarray:
    return vector.reshape(-1, dim, order="F")


def elementwise_dot(
    matrix_1: np.ndarray, matrix_2: np.ndarray, keepdims: bool = False
) -> np.ndarray:
    return (matrix_1 * matrix_2).sum(axis=1, keepdims=keepdims)


@numba.njit
def euclidean_norm_numba(vector: np.ndarray) -> np.ndarray:
    data = (vector**2).sum(axis=-1)
    return np.sqrt(data)


@numba.njit
def get_node_index_numba(node, nodes):
    for i, n in enumerate(nodes):
        if np.sum(np.abs(node - n)) < 0.0001:
            return i
    raise ArgumentError

@numba.njit(inline="always")
def length(edge, nodes):
    return np.sqrt(
        (nodes[edge[0]][0] - nodes[edge[1]][0]) ** 2 + (nodes[edge[0]][1] - nodes[edge[1]][1]) ** 2
    )
