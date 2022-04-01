from typing import Callable

import numpy as np
from numba import njit

from conmech.dataclass.body_properties import (
    StaticBodyProperties,
    TemperatureBodyProperties,
)
from conmech.dataclass.mesh_data import MeshData
from conmech.dataclass.schedule import Schedule
from conmech.solvers.optimization.schur_complement import SchurComplement
from deep_conmech.simulator.dynamics import dynamics_builder_2d, dynamics_builder_3d
from deep_conmech.simulator.mesh.mesh import Mesh


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


class Dynamics(Mesh):
    def __init__(
            self,
            mesh_data: MeshData,
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
            normalize_by_rotation=normalize_by_rotation,
            is_dirichlet=is_dirichlet,
            is_contact=is_contact,
            create_in_subprocess=create_in_subprocess,
        )
        self.body_prop = body_prop
        self.schedule = schedule
        self.with_schur_complement_matrices = with_schur_complement_matrices

        self.reinitialize_matrices()

    def remesh(self):
        super().remesh()
        self.reinitialize_matrices()

    @property
    def time_step(self):
        return self.schedule.time_step

    @property
    def with_temperature(self):
        return isinstance(self.body_prop, TemperatureBodyProperties)

    def reinitialize_matrices(self):
        builder = (
            dynamics_builder_2d.DynamicsBuilder2D()
            if self.dimension == 2
            else dynamics_builder_3d.DynamicsBuilder3D()
        )

        (
            self.element_initial_volume,
            self.const_volume,
            self.ACC,
            self.const_elasticity,
            self.const_viscosity,
            self.C2T,
            self.K,
        ) = builder.build_matrices(
            elements=self.elements,
            nodes=self.normalized_points,
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
