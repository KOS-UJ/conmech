from typing import Callable

import deep_conmech.common.config as config
import numpy as np
from conmech.dataclass.body_coeff import BodyCoeff
from conmech.dataclass.mesh_data import MeshData
from conmech.dataclass.time_data import TimeData
from deep_conmech.simulator.matrices import matrices_2d, matrices_3d
from deep_conmech.simulator.setting.setting_mesh import SettingMesh
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


class SettingMatrices(SettingMesh):
    def __init__(
        self,
        mesh_data: MeshData,
        body_coeff: BodyCoeff,
        time_data: TimeData,
        is_dirichlet: Callable = (lambda _: False),
        is_contact: Callable = (lambda _: True),
        with_schur_complement_matrices: bool = True,
        create_in_subprocess: bool =False,
    ):
        super().__init__(
            mesh_data=mesh_data,
            is_dirichlet=is_dirichlet,
            is_contact=is_contact,
            create_in_subprocess=create_in_subprocess,
        )
        self.body_coeff = body_coeff
        self.time_data = time_data
        self.with_schur_complement_matrices = with_schur_complement_matrices

        self.reinitialize_matrices()

    def remesh(self):
        super().remesh()
        self.reinitialize_matrices()

    @property
    def time_step(self):
        return self.time_data.time_step

    def reinitialize_matrices(self):

        get_edges_features_matrix = (
            lambda *args: matrices_2d.get_edges_features_matrix_numba(*args)
            if self.dim == 2
            else matrices_3d.get_edges_features_matrix_numba(*args)
        )

        get_matrices = (
            lambda *args: matrices_2d.get_matrices(*args)
            if self.dim == 2
            else matrices_3d.get_matrices(*args)
        )

        (
            edges_features_matrix,
            self.element_initial_volume,
        ) = get_edges_features_matrix(self.elements, self.normalized_points)

        # edges_features = get_edges_features_list(
        #    self.edges_number, edges_features_matrix
        # )
        slice_ind = slice(0, self.independent_nodes_count)
        (
            self.C,
            self.B,
            self.AREA,
            self.A_plus_B_times_ts,
            self.A,
            self.ACC,
            self.K,
            self.C2X,
            self.C2Y,
            self.C2T
        ) = get_matrices(
            edges_features_matrix, self.body_coeff, self.time_step, slice_ind,
        )

        if self.with_schur_complement_matrices:
            self.calculate_schur_complement_matrices()

    def calculate_schur_complement_matrices(self):
        p = self.independent_nodes_count
        t = self.boundary_nodes_count
        i = p - t

        C_split = np.array(
            np.split(np.array(np.split(self.C, self.dim, axis=-1)), self.dim, axis=1)
        )
        Ctt = np.moveaxis(C_split[..., :t, :t], 1, 2).reshape(
            self.dim * t, self.dim * t
        )
        self.Cti = np.moveaxis(C_split[..., :t, t:], 1, 2).reshape(
            self.dim * t, self.dim * i
        )
        self.Cit = np.moveaxis(C_split[..., t:, :t], 1, 2).reshape(
            self.dim * i, self.dim * t
        )
        Cii = np.moveaxis(C_split[..., t:, t:], 1, 2).reshape(
            self.dim * i, self.dim * i
        )

        self.CiiINV = np.linalg.inv(Cii)
        CiiINVCit = self.CiiINV @ self.Cit
        CtiCiiINVCit = self.Cti @ CiiINVCit

        self.C_boundary = Ctt - CtiCiiINVCit

    def clear_save(self):
        self.is_contact = None
        self.is_dirichlet = None
        
        self.element_initial_volume = None
        self.A = None
        self.ACC = None
        self.K = None
        self.C2X = None
        self.C2Y = None
        self.C2T = None

        self.B = None
        self.AREA = None
        self.A_plus_B_times_ts = None
        self.Cti = None
        self.Cit = None
        self.CiiINV = None
        self.C_boundary = None
