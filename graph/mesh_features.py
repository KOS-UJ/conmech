import numpy as np
from numba import cuda, jit, njit, prange
from numba.typed import List

import config
import helpers
from mesh import Mesh


@njit
def get_edges_features_list(edges_matrix, edges_features_matrix, edges_features):
    p = len(edges_matrix[0])
    e = 0
    for i in range(p):
        for j in range(p):
            if edges_matrix[i, j]:
                edges_features[e] = edges_features_matrix[i, j]
                e += 1


@njit  # (parallel=True)
def get_edges_features_matrix(cells, cells_points, edges_features_matrix):
    vertices_number = len(cells[0])

    for cell_index in range(len(cells)):  # TODO: prange?
        cell = cells[cell_index]
        cell_points = cells_points[cell_index]

        # TODO: Get rid of repetition (?)
        for i in range(vertices_number):
            i_dPhX, i_dPhY, triangle_area = get_integral_parts(cell_points, i)

            for j in range(vertices_number):
                j_dPhX, j_dPhY, _ = get_integral_parts(cell_points, j)

                area = (i != j) / 6.0
                w11 = i_dPhX * j_dPhX
                w12 = i_dPhX * j_dPhY
                w21 = i_dPhY * j_dPhX
                w22 = i_dPhY * j_dPhY
                u = (1 + (i == j)) / 12.0

                edges_features_matrix[cell[i], cell[j]] += triangle_area * np.array(
                    [area, w11, w12, w21, w22, u]
                )


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
    U = edges_features_matrix[..., 5]

    Z = np.zeros_like(U)
    ACC = DENS * np.vstack((np.hstack((U, Z)), np.hstack((Z, U))))

    A_plus_B_times_ts = A + B * config.TIMESTEP
    C = ACC + A_plus_B_times_ts * config.TIMESTEP

    return C, B, AREA, A_plus_B_times_ts


@njit
def set_diff(data, position, row, i, j):
    vector = data[j] - data[i]
    row[position : position + 2] = vector
    row[position + 2] = np.linalg.norm(vector)
    position += 3


@njit  # (parallel=True)
def get_edges_data(
    edges_data, points_number, edges_matrix, u_old, v_old
):
    p = points_number
    e = 0
    for i in range(p):
        for j in range(p):
            if edges_matrix[i, j]:
                position = 0
                set_diff(u_old, position, edges_data[e], i, j)
                set_diff(v_old, position, edges_data[e], i, j)

                e += 1


class MeshFeatures(Mesh):
    def __init__(
        self, mesh_size, mesh_type, corners, is_adaptive
    ):
        super().__init__(mesh_size, mesh_type, corners, is_adaptive)
        self.reinitialize_matrices()

    #def move(self, u_step):
    #    super().move(u_step)
    #    # self.reinitialize_matrices()

    def reinitialize_matrices(self):
        p = self.points_number
        self.edges_features_matrix = np.zeros((p, p, 6), dtype=np.float)
        self.edges_features = np.zeros((self.edges_number, 6), dtype=np.float)

        get_edges_features_matrix(
            self.cells, self.cells_normalized_points, self.edges_features_matrix
        )

        get_edges_features_list(
            self.edges_matrix, self.edges_features_matrix, self.edges_features
        )

        self.C, self.B, self.AREA, self.A_plus_B_times_ts = get_matrices(
            self.edges_features_matrix,
            MU=config.MU,
            LA=config.LA,
            TH=config.TH,
            ZE=config.ZE,
            DENS=config.DENS,
        )
        a=0

    @property
    def edges_data(self):
        edges_data = np.zeros((self.edges_number, 6), dtype=np.float)
        get_edges_data(
            edges_data,
            self.points_number,
            self.edges_matrix,
            self.input_u_old,
            self.input_v_old
        )
        return edges_data
