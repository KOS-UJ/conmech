from typing import Callable

import deep_conmech.common.config as config
import numba
import numpy as np
from conmech.features.boundaries import Boundaries
from deep_conmech.simulator.setting.setting_mesh import SettingMesh
from numba import njit
from deep_conmech.common import basic_helpers




@njit  # (parallel=True)
def get_edges_features_matrix(elements, nodes):
    nodes_count = len(nodes)
    elements_count, element_size = elements.shape
    dim=element_size-1

    edges_features_matrix = np.zeros((nodes_count, nodes_count, 8), dtype=np.double)
    element_initial_volume = np.zeros(elements_count)

    for element_index in range(elements_count):  # TODO: prange?
        element = elements[element_index]
        element_points = nodes[element]

        # TODO: Get rid of repetition (?)
        for i in range(element_size):
            i_dPhX, i_dPhY, element_volume = get_integral_parts(element_points, i)
            # TODO: Avoid repetition
            element_initial_volume[element_index] = element_volume

            for j in range(element_size):
                j_dPhX, j_dPhY, _ = get_integral_parts(element_points, j)

                area = (i != j) / 6.0
                w11 = i_dPhX * j_dPhX
                w12 = i_dPhX * j_dPhY
                w21 = i_dPhY * j_dPhX
                w22 = i_dPhY * j_dPhY
                u = (1 + (i == j)) / 12.0

                u1 = i_dPhX / 3.0
                u2 = i_dPhY / 3.0

                edges_features_matrix[element[i], element[j]] += element_volume * np.array(
                    [area, w11, w12, w21, w22, u1, u2, u]
                )

    return edges_features_matrix, element_initial_volume




@njit
def get_integral_parts(element_nodes, element_index):
    x_i = element_nodes[element_index % 3]
    x_j1 = element_nodes[(element_index + 1) % 3]
    x_j2 = element_nodes[(element_index + 2) % 3]

    dm = denominator(x_i, x_j1, x_j2)
    triangle_area = np.abs(dm) / 2.0 # = np.abs(dm) / 2.0 = shoelace_area

    y_sub = x_j2[1] - x_j1[1]
    x_sub = x_j1[0] - x_j2[0]

    dPhX = basic_helpers.div_or_zero(y_sub, dm)
    dPhY = basic_helpers.div_or_zero(x_sub, dm)

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



# @njit
def get_edges_features_list(edges_number, edges_features_matrix):
    nodes_count = len(edges_features_matrix[0])
    edges_features = np.zeros((edges_number + nodes_count, 8))  # , dtype=numba.double)
    e = 0
    for i in range(nodes_count):
        for j in range(nodes_count):
            if np.any(edges_features_matrix[i, j]):
                edges_features[e] = edges_features_matrix[i, j]
                e += 1
    return edges_features




@njit
def calculate_constitutive_matrices(W11, W12, W21, W22, MU, LA):
    B11 = (2 * MU + LA) * W11 + MU * W22
    B22 = MU * W11 + (2 * MU + LA) * W22
    B12 = MU * W21 + LA * W12
    B21 = LA * W21 + MU * W12
    return B11, B12, B21, B22


# @njit
def get_matrices(
    edges_features_matrix, MU, LA, TH, ZE, density, time_step, independent_nodes_count
):
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

    ind = slice(0, independent_nodes_count)

    A = np.vstack(
        (
            np.hstack((A11[ind, ind], A12[ind, ind])),
            np.hstack((A21[ind, ind], A22[ind, ind])),
        )
    )
    B = np.vstack(
        (
            np.hstack((B11[ind, ind], B12[ind, ind])),
            np.hstack((B21[ind, ind], B22[ind, ind])),
        )
    )

    U1 = edges_features_matrix[..., 5][ind, ind]
    U2 = edges_features_matrix[..., 6][ind, ind]
    U = edges_features_matrix[..., 7][ind, ind]

    Z = np.zeros_like(U)
    ACC = density * np.vstack((np.hstack((U, Z)), np.hstack((Z, U))))

    A_plus_B_times_ts = A + B * time_step
    C = ACC + A_plus_B_times_ts * time_step

    """
    k11 = 0.5
    k12 = k21 = 0.5
    k22 = 0.5

    c11 = 1.5
    c12 = c21 = 1.5
    c22 = 1.5

    C2X = c11 * U1 + c21 * U2
    C2Y = c12 * U1 + c22 * U2

    # T = (1.0 / TIMESTEP) * k11 * W11 + k12 * W12 + k21 * W21 + k22 * W22
    """

    k11 = k22 = 0.1
    k12 = k21 = 0
    K = k11 * W11 + k12 * W12 + k21 * W21 + k22 * W22

    return C, B, AREA, A_plus_B_times_ts, A, ACC, K


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

        self.reinitialize_matrices()

    def remesh(self):
        super().remesh()
        self.reinitialize_matrices()

    def reinitialize_matrices(self):

        edges_features_matrix, self.element_initial_area = get_edges_features_matrix(
            self.cells, self.normalized_points
        )

        # edges_features = get_edges_features_list(
        #    self.edges_number, edges_features_matrix
        # )

        (
            self.C,
            self.B,
            self.AREA,
            self.A_plus_B_times_ts,
            self.A,
            self.ACC,
            self.K,
        ) = get_matrices(
            edges_features_matrix,
            self.mu,
            self.la,
            self.th,
            self.ze,
            self.density,
            self.time_step,
            self.independent_nodes_count,
        )

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

