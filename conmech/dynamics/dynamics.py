from dataclasses import dataclass
from typing import Callable

import numba
import numpy as np

from conmech.dynamics.factory.dynamics_factory_method import get_dynamics
from conmech.properties.body_properties import (
    StaticBodyProperties,
    TemperatureBodyProperties,
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


class Dynamics(BodyPosition):
    def __init__(
        self,
        mesh_prop: MeshProperties,
        body_prop: StaticBodyProperties,
        schedule: Schedule,
        normalize_by_rotation: bool,
        is_dirichlet: Callable,
        is_contact: Callable,
        create_in_subprocess: bool,
        with_lhs: bool = True,
        with_schur: bool = True,
    ):
        super().__init__(
            mesh_prop=mesh_prop,
            schedule=schedule,
            normalize_by_rotation=normalize_by_rotation,
            is_dirichlet=is_dirichlet,
            is_contact=is_contact,
            create_in_subprocess=create_in_subprocess,
        )
        self.body_prop = body_prop
        self.with_lhs = with_lhs
        self.with_schur = with_schur

        self.element_initial_volume: np.ndarray
        self.volume: np.ndarray
        self.acceleration_operator: np.ndarray
        self.elasticity: np.ndarray
        self.viscosity: np.ndarray
        self.thermal_expansion: np.ndarray
        self.thermal_conductivity: np.ndarray

        self.solver_cache = SolverMatrices()
        self.reinitialize_matrices()

    def remesh(self, is_dirichlet, is_contact, create_in_subprocess):
        super().remesh(is_dirichlet, is_contact, create_in_subprocess)
        self.reinitialize_matrices()

    def reinitialize_matrices(self):
        (
            self.element_initial_volume,
            self.volume,
            self.acceleration_operator,
            self.elasticity,
            self.viscosity,
            self.thermal_expansion,
            self.thermal_conductivity,
        ) = get_dynamics(
            elements=self.elements,
            nodes=self.moved_nodes,
            body_prop=self.body_prop,
            independent_indices=self.independent_indices,
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
                dimension=self.dimension,
                contact_indices=self.contact_indices,
                free_indices=self.free_indices,
            )

            if self.with_temperature:
                i = self.independent_indices
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
                    contact_indices=self.contact_indices,
                    free_indices=self.free_indices,
                )

    @property
    def with_temperature(self):
        return isinstance(self.body_prop, TemperatureBodyProperties)
