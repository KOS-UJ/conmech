from typing import Callable

import deep_conmech.common.config as config
import numba
import numpy as np
from conmech.features.boundaries import Boundaries
from conmech.helpers import nph
from numba import njit


ELEMENT_NODES_COUNT = 3
CONNECTED_EDGES_COUNT = 2
INT_PH = 1 / ELEMENT_NODES_COUNT

@njit  # (parallel=True)
def get_edges_features_matrix_numba(elements, nodes):
    # integral of phi over the element (in 2D: 1/3, in 3D: 1/4)
    nodes_count = len(nodes)
    elements_count, element_size = elements.shape

    edges_features_matrix = np.zeros((8, nodes_count, nodes_count), dtype=np.double)
    element_initial_volume = np.zeros(elements_count)

    for element_index in range(elements_count):  # TODO: prange?
        element = elements[element_index]
        element_points = nodes[element]

        # TODO: Get rid of repetition (?)
        for i in range(element_size):
            i_dPhX, i_dPhY, element_volume = get_integral_parts_numba(element_points, i)
            # TODO: Avoid repetition
            element_initial_volume[element_index] = element_volume

            for j in range(element_size):
                j_dPhX, j_dPhY, _ = get_integral_parts_numba(element_points, j)

                volume = (i != j) * (INT_PH / CONNECTED_EDGES_COUNT)
                # divide by edge count - info about each triangle is "sent" to node via all connected edges
                # (in 2D: 2, in 3D: 3) and summed (by dot product with matrix)
                u = (1 + (i == j)) / 12.0
                # in 3D: divide by 10 or 20, in 2D: divide by 6 or 12

                v1 = INT_PH * j_dPhX
                v2 = INT_PH * j_dPhY

                w11 = i_dPhX * j_dPhX
                w12 = i_dPhX * j_dPhY
                w21 = i_dPhY * j_dPhX
                w22 = i_dPhY * j_dPhY

                edges_features_matrix[
                    :, element[i], element[j]
                ] += element_volume * np.array([volume, u, v1, v2, w11, w12, w21, w22])

    return edges_features_matrix, element_initial_volume


@njit
def get_integral_parts_numba(element_nodes, element_index):
    x_i = element_nodes[element_index]
    x_j1, x_j2 = list(element_nodes[np.arange(ELEMENT_NODES_COUNT) != element_index])

    dm = denominator_numba(x_i, x_j1, x_j2)
    element_volume = np.abs(dm) / 2.0  # = np.abs(dm) / 2.0 = shoelace_area

    y_sub = x_j2[1] - x_j1[1]
    x_sub = x_j1[0] - x_j2[0]

    dPhX = y_sub / dm
    dPhY = x_sub / dm

    return dPhX, dPhY, element_volume


@njit
def shoelace_area_numba(points):
    x = points[:, 0].copy()
    y = points[:, 1].copy()
    area = 0.5 * np.abs(np.dot(x, np.roll(y, 1)) - np.dot(y, np.roll(x, 1)))
    return area

@njit
def denominator_numba(x_i, x_j1, x_j2):
    return (
        x_i[1] * x_j1[0]
        + x_j1[1] * x_j2[0]
        + x_i[0] * x_j2[1]
        - x_i[1] * x_j2[0]
        - x_j2[1] * x_j1[0]
        - x_i[0] * x_j1[1]
    )


def calculate_constitutive_matrices(W11, W12, W21, W22, MU, LA):
    X11 = (2 * MU + LA) * W11 + MU * W22
    X22 = MU * W11 + (2 * MU + LA) * W22
    X12 = MU * W21 + LA * W12
    X21 = LA * W21 + MU * W12
    return np.block([[X11, X12], [X21, X22]])


def create_acceleration(U, density):
    Z = np.zeros_like(U)
    return density * np.block([[U, Z], [Z, U]])


def get_matrices(edges_features_matrix, coefficients, time_step, slice_ind):

    VOL = edges_features_matrix[0]

    U = edges_features_matrix[1][slice_ind, slice_ind]

    ALL_V = [edges_features_matrix[i][slice_ind, slice_ind] for i in range(2, 4)]
    ALL_W = [edges_features_matrix[i][slice_ind, slice_ind] for i in range(4, 8)]

    A = calculate_constitutive_matrices(*ALL_W, coefficients.theta, coefficients.zeta)
    B = calculate_constitutive_matrices(*ALL_W, coefficients.mu, coefficients.lambda_)
    ACC = create_acceleration(U, coefficients.density)

    A_plus_B_times_ts = A + B * time_step
    C = ACC + A_plus_B_times_ts * time_step

    c11 = c22 = 0.5
    c12 = c21 = 0


    V1, V2 = ALL_V
    W11, W12, W21, W22 = ALL_W
    
    C2X = c11 * V1 + c21 * V2
    C2Y = c12 * V1 + c22 * V2

    """
    # T = (1.0 / TIMESTEP) * k11 * W11 + k12 * W12 + k21 * W21 + k22 * W22
    """

    k11 = k22 = 0.1
    k12 = k21 = 0
    K = k11 * W11 + k12 * W12 + k21 * W21 + k22 * W22

    return C, B, VOL, A_plus_B_times_ts, A, ACC, K, C2X, C2Y
