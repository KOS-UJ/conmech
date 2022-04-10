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

    edges_features_matrix, element_initial_volume = factory.get_edges_features_matrix(
        elements, nodes
    )

    i = independent_indices
    const_volume = edges_features_matrix[0][i, i]
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

    elasticity = (
        factory.calculate_constitutive_matrices(W, body_prop.mu, body_prop.lambda_)
        if isinstance(body_prop, StaticBodyProperties)
        else None
    )

    viscosity = (
        factory.calculate_constitutive_matrices(W, body_prop.theta, body_prop.zeta)
        if isinstance(body_prop, DynamicBodyProperties)
        else None
    )

    acceleration_operator = factory.calculate_acceleration(U, body_prop.mass_density)

    if isinstance(body_prop, TemperatureBodyProperties):
        thermal_expansion = factory.calculate_thermal_expansion(V, body_prop.thermal_expansion)
        thermal_conductivity = factory.calculate_thermal_conductivity(
            W, body_prop.thermal_conductivity
        )
    else:
        thermal_expansion = None
        thermal_conductivity = None

    return (
        element_initial_volume,
        const_volume,
        acceleration_operator,
        elasticity,
        viscosity,
        thermal_expansion,
        thermal_conductivity,
    )
