import deep_conmech.common.config as config
import numpy as np
from deep_conmech.simulator.matrices.matrices_2d import get_edges_features_matrix_2d_numba, get_matrices
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
        mesh_type,
        mesh_density_x,
        mesh_density_y,
        scale_x,
        scale_y,
        is_adaptive,
        create_in_subprocess,
        mu=config.MU,
        la=config.LA,
        th=config.TH,
        ze=config.ZE,
        density=config.DENS,
        time_step=config.TIMESTEP,
        reorganize_boundaries=None,
        with_C=True,
    ):
        super().__init__(
            mesh_type,
            mesh_density_x,
            mesh_density_y,
            scale_x,
            scale_y,
            is_adaptive,
            create_in_subprocess,
        )
        self.mu = mu
        self.la = la
        self.th = th
        self.ze = ze
        self.density = density
        self.time_step = time_step

        self.boundaries = None
        self.independent_nodes_count = self.nodes_count
        if reorganize_boundaries is not None:
            reorganize_boundaries()
        self.with_C = with_C

        self.reinitialize_matrices()

    def remesh(self):
        super().remesh()
        self.reinitialize_matrices()

    def reinitialize_matrices(self):

        edges_features_matrix, self.element_initial_area = get_edges_features_matrix_2d_numba(
            self.cells, self.normalized_points
        )

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
        ) = get_matrices(
            edges_features_matrix,
            self.mu,
            self.la,
            self.th,
            self.ze,
            self.density,
            self.time_step,
            slice_ind,
        )

        if self.with_C:
            self.calculate_C()

    def calculate_C(self):
        p = self.independent_nodes_count
        t = self.boundary_nodes_count
        i = p - t

        # self.C = np.array([[i*j for i in range(4)] for j in range(4)])
        C_split = np.array(
            np.split(
                np.array(np.split(self.C, config.DIM, axis=-1)), config.DIM, axis=1
            )
        )
        Ctt = np.moveaxis(C_split[..., :t, :t], 1, 2).reshape(2 * t, 2 * t)
        self.Cti = np.moveaxis(C_split[..., :t, t:], 1, 2).reshape(2 * t, 2 * i)
        self.Cit = np.moveaxis(C_split[..., t:, :t], 1, 2).reshape(2 * i, 2 * t)
        Cii = np.moveaxis(C_split[..., t:, t:], 1, 2).reshape(2 * i, 2 * i)

        self.CiiINV = np.linalg.inv(Cii)
        CiiINVCit = self.CiiINV @ self.Cit
        CtiCiiINVCit = self.Cti @ CiiINVCit

        self.C_boundary = Ctt - CtiCiiINVCit

    def clear_save(self):
        self.element_initial_area = None
        self.A = None
        self.ACC = None
        self.K = None

        self.B = None
        self.AREA = None
        self.A_plus_B_times_ts = None
        self.Cti = None
        self.Cit = None
        self.CiiINV = None
        self.C_boundary = None
