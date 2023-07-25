from dataclasses import dataclass
from typing import Optional

import numba
import numpy as np

from conmech.dynamics.factory.dynamics_factory_method import (
    get_dynamics,
    get_basic_matrices,
    get_factory,
)
from conmech.helpers.schur_complement_functions import calculate_schur_complement_matrices
from conmech.properties.body_properties import (
    TemperatureBodyProperties,
    ElasticRelaxationProperties,
)
from conmech.scene.body_forces import BodyForces


@numba.njit
def get_edges_features_list_numba(edges_number, edges_features_matrix):
    nodes_count = len(edges_features_matrix[0])
    edges_features = np.zeros((edges_number + nodes_count, 8))
    edge_id = 0
    for i in range(nodes_count):
        for j in range(nodes_count):
            if np.any(edges_features_matrix[i, j]):
                edges_features[edge_id] = edges_features_matrix[i, j]
                edge_id += 1
    return edges_features


# TODO: #75
@dataclass
class SolverMatrices:
    def __init__(self):
        self.lhs: np.ndarray
        # TODO: #75 move to schur (careful - some properties are used by net)
        self.lhs_boundary: np.ndarray
        self.free_x_contact: np.ndarray
        self.contact_x_free: np.ndarray
        self.free_x_free_inverted: np.ndarray

        self.lhs_temperature: np.ndarray
        # TODO: #75 move to schur (careful - some properties are used by net)
        self.temperature_boundary: np.ndarray
        self.temperature_free_x_contact: np.ndarray
        self.temperature_contact_x_free: np.ndarray
        self.temperature_free_x_free_inv: np.ndarray


@dataclass
class DynamicsConfiguration:
    normalize_by_rotation: bool = True
    create_in_subprocess: bool = False
    with_lhs: bool = True
    with_schur: bool = True


class Dynamics:
    # pylint: disable=too-many-instance-attributes
    def __init__(
        self,
        body: "Body",
        time_step,
        dynamics_config: DynamicsConfiguration,
    ):
        self.body = body
        self.body.dynamics = self

        self.force = BodyForces(body)
        self.temperature = BodyForces(body, )
        self.time_step = time_step
        self.with_lhs = dynamics_config.with_lhs
        self.with_schur = dynamics_config.with_schur

        self.factory = get_factory(body.mesh.mesh_prop.dimension)
        self.element_initial_volume: np.ndarray
        self.volume_at_nodes: np.ndarray
        self.acceleration_operator: np.ndarray
        self.elasticity: np.ndarray
        self.viscosity: np.ndarray
        self._w_matrix = None
        self.__relaxation: Optional[np.ndarray] = None
        self.__relaxation_tensor: Optional[float] = None
        self.thermal_expansion: np.ndarray
        self.thermal_conductivity: np.ndarray
        self.piezoelectricity: np.ndarray
        self.permittivity: np.ndarray
        self.poisson_operator: np.ndarray

        self.solver_cache = SolverMatrices()
        self.reinitialize_matrices()

    def reinitialize_matrices(self):
        (
            self.element_initial_volume,
            self.volume_at_nodes,
            U,
            V,
            self._w_matrix,
        ) = get_basic_matrices(elements=self.body.mesh.elements, nodes=self.body.mesh.initial_nodes) # + self.displacement_old)
        (
            self.acceleration_operator,
            self.elasticity,
            self.viscosity,
            self.thermal_expansion,
            self.thermal_conductivity,
            self.piezoelectricity,
            self.permittivity,
            self.poisson_operator,
        ) = get_dynamics(
            elements=self.body.mesh.elements, body_prop=self.body.properties, U=U, V=V, W=self._w_matrix
        )

        if not self.with_lhs:
            return

        self.solver_cache.lhs = (
            self.acceleration_operator
            + (self.viscosity + self.elasticity * self.time_step) * self.time_step
        )
        if self.with_schur:
            (
                self.solver_cache.lhs_boundary,
                self.solver_cache.free_x_contact,
                self.solver_cache.contact_x_free,
                self.solver_cache.free_x_free_inverted,
            ) = calculate_schur_complement_matrices(
                matrix=self.solver_cache.lhs,
                dimension=self.body.mesh.dimension,
                contact_indices=self.body.mesh.contact_indices,
                free_indices=self.body.mesh.free_indices,
            )

            if self.with_temperature:
                i = self.body.mesh.independent_indices
                self.solver_cache.lhs_temperature = (
                    1 / self.time_step
                ) * self.acceleration_operator[i, i] + self.thermal_conductivity[i, i]
                (
                    self.solver_cache.temperature_boundary,
                    self.solver_cache.temperature_free_x_contact,
                    self.solver_cache.temperature_contact_x_free,
                    self.solver_cache.temperature_free_x_free_inv,
                ) = calculate_schur_complement_matrices(
                    matrix=self.solver_cache.lhs_temperature,
                    dimension=1,
                    contact_indices=self.body.mesh.contact_indices,
                    free_indices=self.body.mesh.free_indices,
                )

    @property
    def with_temperature(self):
        return isinstance(self.body.properties, TemperatureBodyProperties)

    def relaxation(self, time: float = 0):
        # TODO handle others
        if isinstance(self.body.properties, ElasticRelaxationProperties):
            relaxation_tensor = self.body.properties.relaxation(time)
            if (relaxation_tensor != self.__relaxation_tensor).any():
                self.__relaxation_tensor = relaxation_tensor
                self.__relaxation = self.factory.get_relaxation_tensor(
                    self._w_matrix, relaxation_tensor
                )
        else:
            raise TypeError("There is no relaxation operator!")

        return self.__relaxation
