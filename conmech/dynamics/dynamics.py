from dataclasses import dataclass
from typing import Optional

import numba
import numpy as np

from conmech.dynamics.factory.dynamics_factory_method import (
    get_dynamics,
    get_basic_matrices,
    get_factory,
)
from conmech.mesh.boundaries_description import BoundariesDescription
from conmech.properties.body_properties import (
    TemperatureBodyProperties,
    BodyProperties,
    ElasticRelaxationProperties,
)
from conmech.properties.mesh_properties import MeshProperties
from conmech.properties.schedule import Schedule
from conmech.helpers.schur_complement_functions import calculate_schur_complement_matrices
from conmech.state.body_position import BodyPosition


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


class Dynamics(BodyPosition):
    # TODO: brake inheritance
    # pylint: disable=too-many-instance-attributes
    def __init__(
        self,
        mesh_prop: MeshProperties,
        body_prop: BodyProperties,
        schedule: Schedule,
        dynamics_config: DynamicsConfiguration,
        boundaries_description: BoundariesDescription,
    ):
        super().__init__(
            mesh_prop=mesh_prop,
            schedule=schedule,
            normalize_by_rotation=dynamics_config.normalize_by_rotation,
            boundaries_description=boundaries_description,
            create_in_subprocess=dynamics_config.create_in_subprocess,
        )
        self.body_prop = body_prop
        self.with_lhs = dynamics_config.with_lhs
        self.with_schur = dynamics_config.with_schur

        self.factory = get_factory(mesh_prop.dimension)
        self.element_initial_volume: np.ndarray
        self.volume_at_nodes: np.ndarray
        self.acceleration_operator: np.ndarray
        self.elasticity: np.ndarray
        self.viscosity: np.ndarray
        self._w_matrix: np.ndarray = None
        self._local_stifness_matrices: np.ndarray = None
        self.__relaxation: Optional[np.ndarray] = None
        self.__relaxation_tensor: Optional[float] = None
        self.thermal_expansion: np.ndarray
        self.thermal_conductivity: np.ndarray
        self.piezoelectricity: np.ndarray
        self.permittivity: np.ndarray
        self.poisson_operator: np.ndarray

        self.solver_cache = SolverMatrices()
        self.reinitialize_matrices()

    def remesh(self, boundaries_description, create_in_subprocess):
        # For some reason pylint don't see that Dynamics(BodyPosition) has mesh
        # pylint: disable=no-member
        super().mesh.remesh(boundaries_description, create_in_subprocess)
        self.reinitialize_matrices()

    def reinitialize_matrices(self, elements_density: Optional[np.ndarray] = None):
        (
            self.element_initial_volume,
            self.volume_at_nodes,
            U,
            V,
            self._w_matrix,
            self._local_stifness_matrices,
        ) = get_basic_matrices(elements=self.mesh.elements, nodes=self.moved_nodes)

        if elements_density is not None:
            self._w_matrix = self.asembly_w_matrix_with_density(elements_density)

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
            elements=self.mesh.elements, body_prop=self.body_prop, U=U, V=V, W=self._w_matrix
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
                dimension=self.mesh.dimension,
                contact_indices=self.mesh.contact_indices,
                free_indices=self.mesh.free_indices,
            )

            if self.with_temperature:
                i = self.mesh.independent_indices
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
                    contact_indices=self.mesh.contact_indices,
                    free_indices=self.mesh.free_indices,
                )

    def asembly_w_matrix_with_density(self, elements_density: np.ndarray):
        w_matrix = np.zeros_like(self._w_matrix)
        for element_index, element in enumerate(self.mesh.elements):
            for i, global_i in enumerate(element):
                for j, global_j in enumerate(element):
                    w_matrix[:, :, global_i, global_j] += (
                        elements_density[element_index]
                        * self._local_stifness_matrices[:, :, element_index, i, j]
                    )
        return w_matrix

    @property
    def with_temperature(self):
        return isinstance(self.body_prop, TemperatureBodyProperties)

    def relaxation(self, time: float = 0):
        # TODO handle others
        if isinstance(self.body_prop, ElasticRelaxationProperties):
            relaxation_tensor = self.body_prop.relaxation(time)
            if (relaxation_tensor != self.__relaxation_tensor).any():
                self.__relaxation_tensor = relaxation_tensor
                self.__relaxation = self.factory.get_relaxation_tensor(
                    self._w_matrix, relaxation_tensor
                )
        else:
            raise TypeError("There is no relaxation operator!")

        return self.__relaxation
