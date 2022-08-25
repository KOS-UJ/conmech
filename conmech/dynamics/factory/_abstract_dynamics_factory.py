from ctypes import ArgumentError
from typing import Tuple
import numpy as np
import numba


@numba.njit
def get_coo_sparse_data_numba(keys, values):
    size = len(values)
    if size < 0:
        raise ArgumentError
    feature_matrix_count = len(values[0])
    row = np.zeros(size, dtype=np.int64)
    col = np.zeros(size, dtype=np.int64)
    data = np.zeros((feature_matrix_count, size), dtype=np.float64)
    for index in range(size):
        row[index], col[index] = keys[index]
        data[:, index] = values[index]
        index += 1
    return row, col, data

    
class AbstractDynamicsFactory:
    @property
    def dimension(self) -> int:
        raise NotImplementedError()

    def get_edges_features_dictionary(self, elements, nodes) -> Tuple:
        raise NotImplementedError()

    def calculate_constitutive_matrices(self, W, mu, lambda_):
        raise NotImplementedError()

    def calculate_acceleration(self, U, density):
        raise NotImplementedError()

    def calculate_thermal_expansion(self, V, coeff):
        raise NotImplementedError()

    def calculate_thermal_conductivity(self, W, coeff):
        raise NotImplementedError()
