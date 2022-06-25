from ctypes import ArgumentError

import numba
import numba.typed
import numpy as np
import scipy.sparse

from conmech.dynamics.factory._dynamics_factory_2d import DynamicsFactory2D
from conmech.dynamics.factory._dynamics_factory_3d import DynamicsFactory3D
from conmech.helpers import jxh, nph
from conmech.properties.body_properties import (
    DynamicBodyProperties,
    StaticBodyProperties,
    TemperatureBodyProperties,
)


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

    # edges_features_matrix_OLD, element_initial_volume_OLD = factory.get_edges_features_matrix_OLD(
    #     elements, nodes
    # )

    edges_features_dict, element_initial_volume = factory.get_edges_features_dictionary(
        elements, nodes
    )
    edges_features_matrix = to_edges_features_matrix(
        edges_features_dict=edges_features_dict, nodes_count=len(nodes)
    )
    for i in range(1, len(edges_features_matrix)):
        edges_features_matrix[i] = jxh.slice(edges_features_matrix[i], independent_indices)
    # [np.allclose(edges_features_matrix[k].toarray()[i, i], edges_features_matrix_OLD[k]) for k in range(len(edges_features_matrix))]
    volume_at_nodes_sparse = jxh.to_jax_sparse(edges_features_matrix[0])
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

    elasticity_sparse = jxh.to_jax_sparse(
        factory.calculate_constitutive_matrices(W, body_prop.mu, body_prop.lambda_)
        if isinstance(body_prop, StaticBodyProperties)
        else None
    )

    viscosity_sparse = jxh.to_jax_sparse(
        factory.calculate_constitutive_matrices(W, body_prop.theta, body_prop.zeta)
        if isinstance(body_prop, DynamicBodyProperties)
        else None
    )

    acceleration_operator_sparse = jxh.to_jax_sparse(
        factory.calculate_acceleration(U, body_prop.mass_density)
    )

    if isinstance(body_prop, TemperatureBodyProperties):
        thermal_expansion_sparse = jxh.to_jax_sparse(
            factory.calculate_thermal_expansion(V, body_prop.thermal_expansion)
        )
        thermal_conductivity_sparse = jxh.to_jax_sparse(
            factory.calculate_thermal_conductivity(W, body_prop.thermal_conductivity)
        )
    else:
        thermal_expansion_sparse = None
        thermal_conductivity_sparse = None

    return (
        element_initial_volume,
        volume_at_nodes_sparse,
        acceleration_operator_sparse,
        elasticity_sparse,
        viscosity_sparse,
        thermal_expansion_sparse,
        thermal_conductivity_sparse,
    )
