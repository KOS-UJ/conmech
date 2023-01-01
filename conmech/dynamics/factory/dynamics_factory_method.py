from dataclasses import dataclass

import jax.experimental.sparse
import numpy as np
import scipy.sparse

from conmech.dynamics.factory._abstract_dynamics_factory import (
    get_coo_sparse_data_numba,
)
from conmech.dynamics.factory._dynamics_factory_2d import DynamicsFactory2D
from conmech.dynamics.factory._dynamics_factory_3d import DynamicsFactory3D
from conmech.helpers import jxh
from conmech.properties.body_properties import (
    PiezoelectricBodyProperties,
    StaticBodyProperties,
    TemperatureBodyProperties,
    TimeDependentBodyProperties,
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

        self.piezoelectricity: scipy.sparse.csr_matrix
        self.permittivity: scipy.sparse.csr_matrix

        self.dx_big: np.ndarray
        self.dx_big_jax: jax.experimental.sparse.BCOO
        self.volume_at_nodes_jax: jax.experimental.sparse.BCOO
        self.acceleration_operator_jax: jax.experimental.sparse.BCOO

    def initialize_sparse_jax(self):
        self.dx_big_jax = jxh.to_jax_sparse(self.dx_big)
        self.volume_at_nodes_jax = jxh.to_jax_sparse(self.volume_at_nodes)
        self.acceleration_operator_jax = jxh.to_jax_sparse(self.acceleration_operator)


def to_edges_features_matrix(edges_features_dict: dict, nodes_count: int):
    keys = np.array(list(edges_features_dict.keys()), dtype=np.int64)
    values = np.array(list(edges_features_dict.values()), dtype=np.float64)
    row, col, data = get_coo_sparse_data_numba(keys=keys, values=values)
    shape = (nodes_count, nodes_count)
    edges_features_matrix = [
        scipy.sparse.coo_matrix((i, (row, col)), shape=shape).tocsr() for i in data
    ]
    return edges_features_matrix


def get_dynamics(
    elements: np.ndarray,
    nodes: np.ndarray,
    body_prop: StaticBodyProperties,
    independent_indices: slice,
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
    result.dx_big = factory.to_dx_matrix(
        dx_dict, elements_count=len(nodes), nodes_count=len(elements)
    )

    # Volumeie calculated also for Dirichletnodes, then their influence is removed in lhs for jax
    edges_features_matrix[0] = edges_features_matrix[0].tocsr()
    # Remove Dirichlet nodes
    for i in range(1, len(edges_features_matrix)):
        edges_features_matrix[i] = edges_features_matrix[i].tocsr()[
            independent_indices, independent_indices
        ]

    result.volume_at_nodes = edges_features_matrix[0]
    U = edges_features_matrix[1]

    result.acceleration_operator = factory.calculate_acceleration(U, body_prop.mass_density)

    # return result

    V = np.asarray([edges_features_matrix[2 + j] for j in range(factory.dimension)])
    W = np.asarray(
        [
            [
                edges_features_matrix[2 + factory.dimension * (k + 1) + j]
                for j in range(factory.dimension)
            ]
            for k in range(factory.dimension)
        ]
    )

    result.elasticity = (
        factory.calculate_constitutive_matrices(W, body_prop.mu, body_prop.lambda_)
        if isinstance(body_prop, StaticBodyProperties)
        else None
    )

    result.viscosity = (
        factory.calculate_constitutive_matrices(W, body_prop.theta, body_prop.zeta)
        if isinstance(body_prop, TimeDependentBodyProperties)
        else None
    )

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

    if isinstance(body_prop, PiezoelectricBodyProperties):
        result.piezoelectricity = factory.get_piezoelectric_tensor(W, body_prop.piezoelectricity)
        result.permittivity = factory.get_permittivity_tensor(W, body_prop.permittivity)
    else:
        result.piezoelectricity = None
        result.permittivity = None

    result.initialize_sparse_jax()
    return result
