from typing import Callable

import numpy as np
from conmech.dataclass.body_properties import BodyProperties
from conmech.dataclass.mesh_data import MeshData
from conmech.dataclass.schedule import Schedule
from conmech.solvers.optimization.schur_complement import SchurComplement
from deep_conmech.simulator.matrices import matrices_2d, matrices_3d
from deep_conmech.simulator.setting.mesh import Mesh
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


class SettingMatrices(Mesh):
    def __init__(
        self,
        mesh_data: MeshData,
        body_prop: BodyProperties,
        schedule: Schedule,
        is_dirichlet: Callable = (lambda _: False),
        is_contact: Callable = (lambda _: True),
        with_schur_complement_matrices: bool = True,
        create_in_subprocess: bool = False,
    ):
        super().__init__(
            mesh_data=mesh_data,
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

    def reinitialize_matrices(self):
        get_edges_features_matrix = (
            lambda *args: matrices_2d.get_edges_features_matrix_numba(*args)
            if self.dimension == 2
            else matrices_3d.get_edges_features_matrix_numba(*args)
        )

        get_matrices = (
            lambda *args: matrices_2d.get_matrices(*args)
            if self.dimension == 2
            else matrices_3d.get_matrices(*args)
        )

        (
            edges_features_matrix,
            self.element_initial_volume,
        ) = get_edges_features_matrix(self.elements, self.normalized_points)

        (self.VOL, self.ACC, self.A, self.B, self.C2T, self.K) = get_matrices(
            edges_features_matrix, self.body_prop, self.independent_indices,
        )

        if self.with_schur_complement_matrices:
            self.A_plus_B_times_ts = self.A + self.B * self.time_step
            self.C = self.ACC + self.A_plus_B_times_ts * self.time_step
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

    def clear_save(self):
        self.is_contact = None
        self.is_dirichlet = None

        self.element_initial_volume = None
        self.A = None
        self.ACC = None
        self.K = None
        self.C2T = None

        self.B = None
        self.VOL = None
        self.A_plus_B_times_ts = None

        self.C_boundary = None
        self.free_x_contact = None
        self.contact_x_free = None
        self.free_x_free_inverted = None

        self.T_boundary = None
        self.T_free_x_contact = None
        self.T_contact_x_free = None
        self.T_free_x_free_inverted = None
