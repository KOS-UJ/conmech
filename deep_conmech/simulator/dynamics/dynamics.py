from typing import Callable

import numpy as np
from conmech.mesh.mesh_properties import MeshProperties
from conmech.properties.body_properties import (StaticBodyProperties,
                                                TemperatureBodyProperties)
from conmech.properties.schedule import Schedule
from conmech.solvers.optimization.schur_complement import SchurComplement
from deep_conmech.simulator.dynamics.body_position import BodyPosition
from deep_conmech.simulator.dynamics.factory.dynamics_factory_method import \
    get_dynamics
from numba import njit


@njit
def get_edges_features_list_numba(edges_number, edges_features_matrix):
    nodes_count = len(edges_features_matrix[0])
    edges_features = np.zeros((edges_number + nodes_count, 8))  # , dtype=numba.double)
    e = 0
    for i in range(nodes_count):
        for j in range(nodes_count):
            if np.any(edges_features_matrix[i, j]):
                edges_features[e] = edges_features_matrix[i, j]
                e += 1
    return edges_features



class Dynamics(BodyPosition):
    def __init__(
            self,
            mesh_data: MeshProperties,
            body_prop: StaticBodyProperties,
            schedule: Schedule,
            normalize_by_rotation: bool,
            is_dirichlet: Callable = (lambda _: False),
            is_contact: Callable = (lambda _: True),
            with_schur_complement_matrices: bool = True,
            create_in_subprocess: bool = False,
    ):
        super().__init__(
            mesh_data=mesh_data,
            schedule=schedule,
            normalize_by_rotation=normalize_by_rotation,
            is_dirichlet=is_dirichlet,
            is_contact=is_contact,
            create_in_subprocess=create_in_subprocess,
        )
        self.body_prop = body_prop
        self.with_schur_complement_matrices = with_schur_complement_matrices

        self.element_initial_volume:np.ndarray
        self.const_volume:np.ndarray
        self.ACC:np.ndarray
        self.const_elasticity:np.ndarray
        self.const_viscosity:np.ndarray
        self.C2T:np.ndarray
        self.K:np.ndarray

        self.C:np.ndarray
        self.C_boundary:np.ndarray
        self.free_x_contact:np.ndarray
        self.contact_x_free:np.ndarray
        self.free_x_free_inverted:np.ndarray

        self.T:np.ndarray
        self.T_boundary:np.ndarray
        self.T_free_x_contact:np.ndarray
        self.T_contact_x_free:np.ndarray
        self.T_free_x_free_inverted:np.ndarray

        self.reinitialize_matrices()


    def remesh(self):
        super().remesh()
        self.reinitialize_matrices()


    def reinitialize_matrices(self):
        (
            self.element_initial_volume,
            self.const_volume,
            self.ACC,
            self.const_elasticity,
            self.const_viscosity,
            self.C2T,
            self.K,
        ) = get_dynamics(
            elements=self.elements,
            nodes=self.moved_nodes,
            body_prop=self.body_prop,
            independent_indices=self.independent_indices,
        )

        if self.with_schur_complement_matrices:
            self.C = (
                    self.ACC
                    + (self.const_viscosity + self.const_elasticity * self.time_step)
                    * self.time_step
            )
            (
                self.C_boundary,
                self.free_x_contact,
                self.contact_x_free,
                self.free_x_free_inverted,
            ) = SchurComplement.calculate_schur_complement_matrices(
                matrix=self.C,
                dimension=self.dimension,
                contact_indices=self.contact_indices,
                free_indices=self.free_indices,
            )

            if self.with_temperature:
                i = self.independent_indices
                self.T = (1 / self.time_step) * self.ACC[i, i] + self.K[i, i]
                (
                    self.T_boundary,
                    self.T_free_x_contact,
                    self.T_contact_x_free,
                    self.T_free_x_free_inverted,
                ) = SchurComplement.calculate_schur_complement_matrices(
                    matrix=self.T,
                    dimension=1,
                    contact_indices=self.contact_indices,
                    free_indices=self.free_indices,
                )


    @property
    def with_temperature(self):
        return isinstance(self.body_prop, TemperatureBodyProperties)
