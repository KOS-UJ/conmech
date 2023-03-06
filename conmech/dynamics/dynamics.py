from dataclasses import dataclass

import numba
import numpy as np

from conmech.dynamics.factory.dynamics_factory_method import get_dynamics
from conmech.mesh.boundaries_description import BoundariesDescription
from conmech.properties.body_properties import (
    TemperatureBodyProperties,
    BodyProperties,
)
from conmech.properties.mesh_properties import MeshProperties
from conmech.properties.schedule import Schedule
from conmech.solvers.optimization.schur_complement import SchurComplement
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

        self.element_initial_volume: np.ndarray
        self.volume_at_nodes: np.ndarray
        self.acceleration_operator: np.ndarray
        self.elasticity: np.ndarray
        self.viscosity: np.ndarray
        self.long_memory: np.ndarray
        self.thermal_expansion: np.ndarray
        self.thermal_conductivity: np.ndarray
        self.piezoelectricity: np.ndarray
        self.permittivity: np.ndarray

        self.solver_cache = SolverMatrices()
        self.reinitialize_matrices()

    def remesh(self, boundaries_description, create_in_subprocess):
        # For some reason pylint don't see that Dynamics(BodyPosition) has mesh
        # pylint: disable=no-member
        super().mesh.remesh(boundaries_description, create_in_subprocess)
        self.reinitialize_matrices()

    def reinitialize_matrices(self):
        (
            self.element_initial_volume,
            self.volume_at_nodes,
            self.acceleration_operator,
            self.elasticity,
            self.viscosity,
            self.long_memory,
            self.thermal_expansion,
            self.thermal_conductivity,
            self.piezoelectricity,
            self.permittivity,
        ) = get_dynamics(
            elements=self.mesh.elements, nodes=self.moved_nodes, body_prop=self.body_prop
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
            ) = SchurComplement.calculate_schur_complement_matrices(
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
                ) = SchurComplement.calculate_schur_complement_matrices(
                    matrix=self.solver_cache.lhs_temperature,
                    dimension=1,
                    contact_indices=self.mesh.contact_indices,
                    free_indices=self.mesh.free_indices,
                )

    @property
    def with_temperature(self):
        return isinstance(self.body_prop, TemperatureBodyProperties)
