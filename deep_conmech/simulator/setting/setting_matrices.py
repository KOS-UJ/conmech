import deep_conmech.common.config as config
import numpy as np
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
        mesh_type,
        mesh_density_x,
        mesh_density_y=None,
        scale_x=None,
        scale_y=None,
        is_adaptive=False,
        create_in_subprocess=False,
        mu_coef=config.MU,
        la_coef=config.LA,
        th_coef=config.TH,
        ze_coef=config.ZE,
        density=config.DENS,
        time_step=config.TIMESTEP,
        is_dirichlet=(lambda _: False),
        is_contact=(lambda _: True),
        with_schur_complement_matrices=True,
    ):
        super().__init__(
            mesh_type=mesh_type,
            mesh_density_x=mesh_density_x,
            mesh_density_y=mesh_density_y,
            scale_x=scale_x,
            scale_y=scale_y,
            is_adaptive=is_adaptive,
            create_in_subprocess=create_in_subprocess,
            is_dirichlet=is_dirichlet,
            is_contact=is_contact,
        )
        self.mu = mu_coef
        self.la = la_coef
        self.th = th_coef
        self.ze = ze_coef
        self.density = density
        self.time_step = time_step
        self.with_schur_complement_matrices = with_schur_complement_matrices

        self.reinitialize_matrices()

    def remesh(self):
        super().remesh()
        self.reinitialize_matrices()

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
        self.element_initial_volume = None
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
