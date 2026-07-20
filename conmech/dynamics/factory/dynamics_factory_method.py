# CONMECH @ Jagiellonian University in Kraków
#
# Copyright (C) 2022-2026  Piotr Bartman-Szwarc <piotr.bartman@uj.edu.pl>
#
# This program is free software; you can redistribute it and/or
# modify it under the terms of the GNU General Public License
# as published by the Free Software Foundation; either version 3
# of the License, or (at your option) any later version.
#
# This program is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
# GNU General Public License for more details.
#
# You should have received a copy of the GNU General Public License
# along with this program; if not, write to the Free Software
# Foundation, Inc., 51 Franklin Street, Fifth Floor, Boston, MA  02110-1301,
# USA.
import numpy as np
from conmech.helpers.assembly import split_features
from conmech.dynamics.factory._dynamics_factory_2d import DynamicsFactory2D
from conmech.dynamics.factory._dynamics_factory_3d import DynamicsFactory3D
from conmech.properties.body_properties import (
    ViscoelasticProperties,
    ElasticProperties,
    TemperatureBodyProperties,
    PiezoelectricBodyProperties,
    BodyProperties,
    MembraneProperties,
)


def get_factory(dimension: int):
    if dimension == 2:
        factory = DynamicsFactory2D()
    elif dimension == 3:
        factory = DynamicsFactory3D()
    else:
        raise NotImplementedError()

    return factory


def get_basic_matrices(elements: np.ndarray, nodes: np.ndarray):
    dimension = len(elements[0]) - 1
    factory = get_factory(dimension)

    (
        edges_features_matrix,
        element_initial_volume,
        local_stifness_matrices,
    ) = factory.get_edges_features_matrix(elements, nodes)

    volume_at_nodes, U, V, W = split_features(edges_features_matrix, factory.dimension)
    return element_initial_volume, volume_at_nodes, U, V, W, local_stifness_matrices


def get_dynamics(elements: np.ndarray, body_prop: BodyProperties, U, V, W):
    dimension = len(elements[0]) - 1
    factory = get_factory(dimension)

    acceleration_operator = factory.calculate_acceleration(U, body_prop.mass_density)

    if isinstance(body_prop, MembraneProperties):
        poisson_operator = factory.calculate_poisson_matrix(W, body_prop.propagation)
    else:
        poisson_operator = None

    if isinstance(body_prop, ElasticProperties):
        elasticity = factory.calculate_constitutive_matrices(W, body_prop.mu, body_prop.lambda_)
    else:
        elasticity = None

    if isinstance(body_prop, ViscoelasticProperties):
        viscosity = factory.calculate_constitutive_matrices(W, body_prop.theta, body_prop.zeta)
    else:
        viscosity = None

    if isinstance(body_prop, TemperatureBodyProperties):
        thermal_expansion = factory.calculate_thermal_expansion(V, body_prop.thermal_expansion)
        thermal_conductivity = factory.calculate_thermal_conductivity(
            W, body_prop.thermal_conductivity
        )
    else:
        thermal_expansion = None
        thermal_conductivity = None

    if isinstance(body_prop, PiezoelectricBodyProperties):
        piezoelectricity = factory.get_piezoelectric_tensor(W, body_prop.piezoelectricity)
        permittivity = factory.get_permittivity_tensor(W, body_prop.permittivity)
    else:
        piezoelectricity = None
        permittivity = None

    return (
        acceleration_operator,
        elasticity,
        viscosity,
        thermal_expansion,
        thermal_conductivity,
        piezoelectricity,
        permittivity,
        poisson_operator,
    )
