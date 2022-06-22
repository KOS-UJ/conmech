import jax.experimental.sparse
import numpy as np

from conmech.dynamics.factory._dynamics_factory_2d import DynamicsFactory2D
from conmech.dynamics.factory._dynamics_factory_3d import DynamicsFactory3D
from conmech.properties.body_properties import (
    DynamicBodyProperties,
    StaticBodyProperties,
    TemperatureBodyProperties,
)


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

    to_sparse = jax.experimental.sparse.BCOO.fromdense  # nph.to_sparse
    edges_features_matrix, element_initial_volume = factory.get_edges_features_matrix(
        elements, nodes
    )

    i = independent_indices
    volume_at_nodes_sparse = to_sparse(edges_features_matrix[0])  # [i, i]
    U = edges_features_matrix[1][i, i]

    V = np.asarray([edges_features_matrix[2 + j][i, i] for j in range(factory.dimension)])
    W = np.asarray(
        [
            [
                edges_features_matrix[2 + factory.dimension * (k + 1) + j][i, i]
                for j in range(factory.dimension)
            ]
            for k in range(factory.dimension)
        ]
    )

    elasticity_sparse = to_sparse(
        factory.calculate_constitutive_matrices(W, body_prop.mu, body_prop.lambda_)
        if isinstance(body_prop, StaticBodyProperties)
        else None
    )

    viscosity_sparse = to_sparse(
        factory.calculate_constitutive_matrices(W, body_prop.theta, body_prop.zeta)
        if isinstance(body_prop, DynamicBodyProperties)
        else None
    )

    acceleration_operator_sparse = to_sparse(
        factory.calculate_acceleration(U, body_prop.mass_density)
    )

    if isinstance(body_prop, TemperatureBodyProperties):
        thermal_expansion_sparse = to_sparse(
            factory.calculate_thermal_expansion(V, body_prop.thermal_expansion)
        )
        thermal_conductivity_sparse = to_sparse(
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
