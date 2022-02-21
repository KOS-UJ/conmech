import numpy as np
import numba
from numba import njit
import common.config as config
from simulator.setting.setting_mesh import SettingMesh


# @njit
def get_edges_features_list(edges_number, edges_features_matrix):
    points_number = len(edges_features_matrix[0])
    edges_features = np.zeros(
        (edges_number + points_number, 8)
    )  # , dtype=numba.double)
    e = 0
    for i in range(points_number):
        for j in range(points_number):
            if np.any(edges_features_matrix[i, j]):
                edges_features[e] = edges_features_matrix[i, j]
                e += 1
    return edges_features


@njit  # (parallel=True)
def get_edges_features_matrix(points_number, cells, cells_points):
    edges_features_matrix = np.zeros(
        (points_number, points_number, 8), dtype=numba.double
    )
    cell_vertices_number = len(cells[0])

    for cell_index in range(len(cells)):  # TODO: prange?
        cell = cells[cell_index]
        cell_points = cells_points[cell_index]

        # TODO: Get rid of repetition (?)
        for i in range(cell_vertices_number):
            i_dPhX, i_dPhY, triangle_area = get_integral_parts(cell_points, i)

            for j in range(cell_vertices_number):
                j_dPhX, j_dPhY, _ = get_integral_parts(cell_points, j)

                area = (i != j) / 6.0
                w11 = i_dPhX * j_dPhX
                w12 = i_dPhX * j_dPhY
                w21 = i_dPhY * j_dPhX
                w22 = i_dPhY * j_dPhY
                u = (1 + (i == j)) / 12.0

                u1 = i_dPhX / 3.0
                u2 = i_dPhY / 3.0

                edges_features_matrix[cell[i], cell[j]] += triangle_area * np.array(
                    [area, w11, w12, w21, w22, u1, u2, u]
                )
    return edges_features_matrix


@njit
def get_integral_parts(cell_points, vertex_index):
    x_i = cell_points[vertex_index % 3]
    x_j1 = cell_points[(vertex_index + 1) % 3]
    x_j2 = cell_points[(vertex_index + 2) % 3]

    dm = denominator(x_i, x_j1, x_j2)  # np.abs(dm) / 2 = shoelace_area
    triangle_area = np.abs(dm) / 2.0

    y_sub = x_j2[1] - x_j1[1]
    x_sub = x_j1[0] - x_j2[0]

    dPhX = div_or_zero(y_sub, dm)
    dPhY = div_or_zero(x_sub, dm)

    return dPhX, dPhY, triangle_area


@njit
def shoelace_area(points):
    x = points[:, 0].copy()
    y = points[:, 1].copy()
    area = 0.5 * np.abs(np.dot(x, np.roll(y, 1)) - np.dot(y, np.roll(x, 1)))
    return area


@njit
def denominator(x_i, x_j1, x_j2):
    return (
        x_i[1] * x_j1[0]
        + x_j1[1] * x_j2[0]
        + x_i[0] * x_j2[1]
        - x_i[1] * x_j2[0]
        - x_j2[1] * x_j1[0]
        - x_i[0] * x_j1[1]
    )


@njit
def div_or_zero(value, denominator):
    if denominator != 0:
        return value / denominator
    else:
        return 0.0


@njit
def calculate_constitutive_matrices_with_angle(W11, W12, W21, W22, MU, LA, angle):
    s = np.sin(angle)
    c = np.cos(angle)
    B11 = (MU + LA + MU * c) * W11 + (MU * s) * W12 + (-MU * s) * W21 + (MU * c) * W22
    B12 = (MU * s) * W11 + (MU + LA - MU * c) * W12 + (MU * c) * W21 + (MU * s) * W22
    B21 = (-MU * s) * W11 + (MU * c) * W12 + (MU + LA - MU * c) * W21 + (-MU * s) * W22
    B22 = (MU * c) * W11 + (MU * s) * W12 + (-MU * s) * W21 + (MU + LA + MU * c) * W22
    return B11, B12, B21, B22


@njit
def calculate_constitutive_matrices(W11, W12, W21, W22, MU, LA):
    B11 = (2 * MU + LA) * W11 + MU * W22
    B22 = MU * W11 + (2 * MU + LA) * W22
    B12 = MU * W21 + LA * W12
    B21 = LA * W21 + MU * W12
    return B11, B12, B21, B22


# @njit
def get_matrices(edges_features_matrix, MU, LA, TH, ZE, DENS):
    # move config MU, LA,... out to model
    AREA = edges_features_matrix[..., 0]

    W11 = edges_features_matrix[..., 1]
    W12 = edges_features_matrix[..., 2]
    W21 = edges_features_matrix[..., 3]
    W22 = edges_features_matrix[..., 4]

    A11, A12, A21, A22 = calculate_constitutive_matrices(W11, W12, W21, W22, TH, ZE)
    B11, B12, B21, B22 = calculate_constitutive_matrices(W11, W12, W21, W22, MU, LA)

    # A = np.vstack((np.hstack((c, s)), np.hstack((-s, c))))
    # result = A @ matrix @ A.T

    A = np.vstack((np.hstack((A11, A12)), np.hstack((A21, A22))))
    B = np.vstack((np.hstack((B11, B12)), np.hstack((B21, B22))))

    U1 = edges_features_matrix[..., 5]
    U2 = edges_features_matrix[..., 6]
    U = edges_features_matrix[..., 7]

    Z = np.zeros_like(U)
    ACC = DENS * np.vstack((np.hstack((U, Z)), np.hstack((Z, U))))

    A_plus_B_times_ts = A + B * config.TIMESTEP
    C = ACC + A_plus_B_times_ts * config.TIMESTEP

    k11 = 0.5
    k12 = k21 = 0.5
    k22 = 0.5

    c11 = 1.5
    c12 = c21 = 1.5
    c22 = 1.5

    C2X = c11 * U1 + c21 * U2
    C2Y = c12 * U1 + c22 * U2

    T = (1.0 / config.TIMESTEP) * k11 * W11 + k12 * W12 + k21 * W21 + k22 * W22

    return C, B, AREA, A_plus_B_times_ts




class SettingFeatures(SettingMesh):
    def __init__(
        self, mesh_density, mesh_type, scale, is_adaptive, create_in_subprocess
    ):
        super().__init__(
            mesh_density, mesh_type, scale, is_adaptive, create_in_subprocess
        )
        self.reinitialize_matrices()


    def remesh(self):
        super().remesh()
        self.reinitialize_matrices()


    def reinitialize_matrices(self):
        edges_features_matrix = get_edges_features_matrix(
            self.points_number, self.cells, self.cells_normalized_points
        )

        #edges_features = get_edges_features_list(
        #    self.edges_number, edges_features_matrix
        #)

        self.C, self.B, self.AREA, self.A_plus_B_times_ts = get_matrices(
            edges_features_matrix,
            MU=config.MU,
            LA=config.LA,
            TH=config.TH,
            ZE=config.ZE,
            DENS=config.DENS,
        )

        p = self.points_number
        t = self.boundary_points_count
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
        a = 0

    def clear_save(self):
        self.B = None
        self.AREA = None
        self.A_plus_B_times_ts = None
        self.Cti = None
        self.Cit = None
        self.CiiINV = None
        self.C_boundary = None

