from ctypes import ArgumentError
from dataclasses import dataclass

import cupyx.scipy.sparse
import numba
import numba.typed
import numpy as np
import scipy.sparse

from conmech.dynamics.factory._dynamics_factory_2d import DynamicsFactory2D
from conmech.dynamics.factory._dynamics_factory_3d import DynamicsFactory3D
from conmech.helpers import jxh
from conmech.properties.body_properties import (
    DynamicBodyProperties,
    StaticBodyProperties,
    TemperatureBodyProperties,
)


@dataclass
class ConstMatrices:
    def __init__(self):
        self.element_initial_volume: np.ndarray
        self.volume_at_nodes: scipy.sparse.csr_matrix
        self.acceleration_operator: scipy.sparse.csr_matrix
        self.elasticity: scipy.sparse.csr_matrix
        self.viscosity: scipy.sparse.csr_matrix
        self.thermal_expansion: scipy.sparse.csr_matrix
        self.thermal_conductivity: scipy.sparse.csr_matrix
        self.volume_at_nodes_cp: cupyx.scipy.sparse.csr_matrix
        self.volume_at_nodes_jax: cupyx.scipy.sparse.csr_matrix
        self.acceleration_operator_cp: cupyx.scipy.sparse.csr_matrix
        self.elasticity_cp: cupyx.scipy.sparse.csr_matrix
        self.viscosity_cp: cupyx.scipy.sparse.csr_matrix

    def initialize_sparse(self):
        self.volume_at_nodes_cp = jxh.to_cupy_csr(self.volume_at_nodes)
        self.acceleration_operator_cp = jxh.to_cupy_csr(self.acceleration_operator)
        self.elasticity_cp = jxh.to_cupy_csr(self.elasticity)
        self.viscosity_cp = jxh.to_cupy_csr(self.viscosity)


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


def to_edges_features_matrix(edges_features_dict: dict, nodes_count: int):
    keys = np.array(list(edges_features_dict.keys()))
    values = np.array(list(edges_features_dict.values()))
    row, col, data = get_coo_sparse_data_numba(keys=keys, values=values)
    shape = (nodes_count, nodes_count)
    edges_features_matrix = [scipy.sparse.coo_matrix((i, (row, col)), shape=shape) for i in data]
    return edges_features_matrix


def to_dx_matrix(dx_dict: dict, elements_count: int, nodes_count: int):
    keys = np.array(list(dx_dict.keys()))
    values = np.array(list(dx_dict.values()))
    row, col, data = get_coo_sparse_data_numba(keys=keys, values=values)
    shape = (elements_count, nodes_count)
    dx = scipy.sparse.coo_matrix((data, (row, col)), shape=shape)
    return dx


def get_dynamics(
    elements: np.ndarray,
    nodes: np.ndarray,
    independent_indices: slice,
    body_prop: StaticBodyProperties,
):
    dimension = len(elements[0]) - 1
    if dimension == 2:
        factory = DynamicsFactory2D()
    elif dimension == 3:
        factory = DynamicsFactory3D()
    else:
        raise NotImplementedError()
    result = ConstMatrices()

    (
        edges_features_dict,
        result.element_initial_volume,
        dx_dict,
    ) = factory.get_edges_features_dictionary(elements, nodes)
    edges_features_matrix = to_edges_features_matrix(
        edges_features_dict=edges_features_dict, nodes_count=len(nodes)
    )
    dx = dx_dict  # to_dx_matrix(dx_dict, elements_count=len(nodes), nodes_count=len(elements)).tocsr()

    edges_features_matrix[0] = edges_features_matrix[0].tocsr()
    for i in range(1, len(edges_features_matrix)):
        edges_features_matrix[i] = edges_features_matrix[i].tocsr()[
            independent_indices, independent_indices
        ]

    result.volume_at_nodes = edges_features_matrix[0]
    U = edges_features_matrix[1]

    V = np.asarray([edges_features_matrix[2 + j] for j in range(factory.dimension)])  # [i, i]
    W = np.asarray(
        [
            [
                edges_features_matrix[2 + factory.dimension * (k + 1) + j]
                for j in range(factory.dimension)
            ]
            for k in range(factory.dimension)
        ]
    )  # [i, i]

    result.elasticity_on = (
        factory.calculate_constitutive_matrices(W, body_prop.mu, body_prop.lambda_)
        if isinstance(body_prop, StaticBodyProperties)
        else None
    )

    result.elasticity = 0 * result.elasticity_on

    result.viscosity = 0 * (
        factory.calculate_constitutive_matrices(W, body_prop.theta, body_prop.zeta)
        if isinstance(body_prop, DynamicBodyProperties)
        else None
    )

    result.acceleration_operator = factory.calculate_acceleration(U, body_prop.mass_density)

    if isinstance(body_prop, TemperatureBodyProperties):
        result.thermal_expansion = factory.calculate_thermal_expansion(
            V, body_prop.thermal_expansion
        )

        result.thermal_conductivity = factory.calculate_thermal_conductivity(
            W, body_prop.thermal_conductivity
        )

    else:
        result.thermal_expansion = None
        result.thermal_conductivity = None

    result.dx = dx
    result.initialize_sparse()
    return result
