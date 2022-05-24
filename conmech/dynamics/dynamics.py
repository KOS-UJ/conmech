from dataclasses import dataclass
from typing import Callable

import numba
import numpy as np
from scipy import sparse

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
        self.lhs_sparse: sparse.base.spmatrix
        # TODO: #75 move to schur (careful - some properties are used by net)
        self.lhs_boundary: np.ndarray
        self.free_x_contact: np.ndarray
        self.contact_x_free: np.ndarray
        self.free_x_free_inverted: np.ndarray

        self.lhs_temperature_sparse: sparse.base.spmatrix
        # TODO: #75 move to schur (careful - some properties are used by net)
        self.temperature_boundary: np.ndarray
        self.temperature_free_x_contact: np.ndarray
        self.temperature_contact_x_free: np.ndarray
        self.temperature_free_x_free_inv: np.ndarray

    @property
    def lhs(self):
        return self.lhs_sparse.toarray()

    @property
    def lhs_temperature(self):
        return self.lhs_temperature_sparse.toarray()


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
        body_prop: StaticBodyProperties,
        schedule: Schedule,
        dynamics_config: DynamicsConfiguration,
        is_dirichlet: Callable,
        is_contact: Callable,
    ):
        super().__init__(
            mesh_prop=mesh_prop,
            schedule=schedule,
            normalize_by_rotation=dynamics_config.normalize_by_rotation,
            is_dirichlet=is_dirichlet,
            is_contact=is_contact,
            create_in_subprocess=dynamics_config.create_in_subprocess,
        )
        self.body_prop = body_prop
        self.with_lhs = dynamics_config.with_lhs
        self.with_schur = dynamics_config.with_schur

        self.element_initial_volume: np.ndarray
        self.volume_at_nodes_sparse: sparse.base.spmatrix
        self.acceleration_operator_sparse: sparse.base.spmatrix
        self.elasticity_sparse: sparse.base.spmatrix
        self.viscosity_sparse: sparse.base.spmatrix
        self.thermal_expansion_sparse: sparse.base.spmatrix
        self.thermal_conductivity_sparse: sparse.base.spmatrix

        self.solver_cache = SolverMatrices()
        self.reinitialize_matrices()

    def remesh(self, is_dirichlet, is_contact, create_in_subprocess):
        super().remesh(is_dirichlet, is_contact, create_in_subprocess)
        self.reinitialize_matrices()

    def reinitialize_matrices(self):
        (
            self.element_initial_volume,
            self.volume_at_nodes_sparse,
            self.acceleration_operator_sparse,
            self.elasticity_sparse,
            self.viscosity_sparse,
            self.thermal_expansion_sparse,
            self.thermal_conductivity_sparse,
        ) = get_dynamics(
            elements=self.elements,
            nodes=self.moved_nodes,
            body_prop=self.body_prop,
            independent_indices=self.independent_indices,
        )

        if not self.with_lhs:
            return

        self.solver_cache.lhs_sparse = (
            self.acceleration_operator_sparse
            + (self.viscosity_sparse + self.elasticity_sparse * self.time_step) * self.time_step
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
                self.solver_cache.lhs_temperature_sparse = (
                    1 / self.time_step
                ) * self.acceleration_operator_sparse[i, i] + self.thermal_conductivity_sparse[i, i]
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
    def volume_at_nodes(self):
        return self.volume_at_nodes_sparse.toarray()

    @property
    def acceleration_operator(self):
        return self.acceleration_operator_sparse.toarray()

    @property
    def elasticity(self):
        return self.elasticity_sparse.toarray()

    @property
    def viscosity(self):
        return self.viscosity_sparse.toarray()

    @property
    def thermal_expansion(self):
        return self.thermal_expansion_sparse.toarray()

    @property
    def thermal_conductivity(self):
        return self.thermal_conductivity_sparse.toarray()

    @property
    def with_temperature(self):
        return isinstance(self.body_prop, TemperatureBodyProperties)
