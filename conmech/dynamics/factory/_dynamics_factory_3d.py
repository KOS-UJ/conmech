import numpy as np
from numba import njit

from conmech.dynamics.factory._abstract_dynamics_factory import AbstractDynamicsFactory

DIMENSION = 3
ELEMENT_NODES_COUNT = 4
CONNECTED_EDGES_COUNT = 3
INT_PH = 1 / ELEMENT_NODES_COUNT
U_DIVIDER = 20
FEATURE_MATRIX_COUNT = 2 + DIMENSION + DIMENSION ** 2
VOLUME_DIVIDER = 6


@njit
def get_edges_features_matrix_numba(elements, nodes):
    # integral of phi over the element (in 2D: 1/3, in 3D: 1/4)
    nodes_count = len(nodes)
    elements_count, element_size = elements.shape

    edges_features_matrix = np.zeros((FEATURE_MATRIX_COUNT, nodes_count, nodes_count),
                                     dtype=np.double)
    element_initial_volume = np.zeros(elements_count)

    for element_index in range(elements_count):  # TODO: #65 prange?
        element = elements[element_index]
        element_points = nodes[element]

        # TODO: #65 Get rid of repetition (?)
        for i in range(element_size):
            i_dPhX, i_dPhY, i_dPhZ, element_volume = get_integral_parts_numba(element_points, i)
            # TODO: #65 Avoid repetition
            element_initial_volume[element_index] = element_volume

            for j in range(element_size):
                j_dPhX, j_dPhY, j_dPhZ, _ = get_integral_parts_numba(element_points, j)

                volume = (i != j) * (INT_PH / CONNECTED_EDGES_COUNT)
                # divide by edge count - info about each triangle is "sent" to node via all connected edges
                # (in 2D: 2, in 3D: 3) and summed (by dot product with matrix)
                u = (1 + (i == j)) / U_DIVIDER
                # in 3D: divide by 10 or 20, in 2D: divide by 6 or 12

                v1 = INT_PH * j_dPhX
                v2 = INT_PH * j_dPhY
                v3 = INT_PH * j_dPhZ

                w11 = i_dPhX * j_dPhX
                w12 = i_dPhX * j_dPhY
                w13 = i_dPhX * j_dPhZ

                w21 = i_dPhY * j_dPhX
                w22 = i_dPhY * j_dPhY
                w23 = i_dPhY * j_dPhZ

                w31 = i_dPhZ * j_dPhX
                w32 = i_dPhZ * j_dPhY
                w33 = i_dPhZ * j_dPhZ

                edges_features_matrix[:, element[i], element[j]] += element_volume * np.array([
                    volume,
                    u,
                    v1, v2, v3,
                    w11, w12, w13,
                    w21, w22, w23,
                    w31, w32, w33,
                ])

    return edges_features_matrix, element_initial_volume


@njit
def get_integral_parts_numba(element_nodes, element_index):
    x_i = element_nodes[element_index]
    x_j1, x_j2, x_j3 = list(
        element_nodes[np.arange(ELEMENT_NODES_COUNT) != element_index]
    )

    dm = denominator_numba(x_i, x_j1, x_j2, x_j3)
    element_volume = np.abs(dm) / VOLUME_DIVIDER

    x_sub = (
            x_j1[2] * x_j2[1]
            - x_j1[1] * x_j2[2]
            - x_j1[2] * x_j3[1]
            + x_j2[2] * x_j3[1]
            + x_j1[1] * x_j3[2]
            - x_j2[1] * x_j3[2]
    )
    y_sub = (
            x_j1[0] * x_j2[2]
            - x_j1[2] * x_j2[0]
            + x_j1[2] * x_j3[0]
            - x_j2[2] * x_j3[0]
            - x_j1[0] * x_j3[2]
            + x_j2[0] * x_j3[2]
    )
    z_sub = (
            x_j1[1] * x_j2[0]
            - x_j1[0] * x_j2[1]
            - x_j1[1] * x_j3[0]
            + x_j2[1] * x_j3[0]
            + x_j1[0] * x_j3[1]
            - x_j2[0] * x_j3[1]
    )

    dPhX = x_sub / dm
    dPhY = y_sub / dm
    dPhZ = z_sub / dm

    return dPhX, dPhY, dPhZ, element_volume


@njit
def denominator_numba(x_i, x_j1, x_j2, x_j3):
    return (
            x_i[2] * x_j1[1] * x_j2[0]
            - x_i[1] * x_j1[2] * x_j2[0]
            - x_i[2] * x_j1[0] * x_j2[1]
            + x_i[0] * x_j1[2] * x_j2[1]
            + x_i[1] * x_j1[0] * x_j2[2]
            - x_i[0] * x_j1[1] * x_j2[2]
            - x_i[2] * x_j1[1] * x_j3[0]
            + x_i[1] * x_j1[2] * x_j3[0]
            + x_i[2] * x_j2[1] * x_j3[0]
            - x_j1[2] * x_j2[1] * x_j3[0]
            - x_i[1] * x_j2[2] * x_j3[0]
            + x_j1[1] * x_j2[2] * x_j3[0]
            + x_i[2] * x_j1[0] * x_j3[1]
            - x_i[0] * x_j1[2] * x_j3[1]
            - x_i[2] * x_j2[0] * x_j3[1]
            + x_j1[2] * x_j2[0] * x_j3[1]
            + x_i[0] * x_j2[2] * x_j3[1]
            - x_j1[0] * x_j2[2] * x_j3[1]
            - x_i[1] * x_j1[0] * x_j3[2]
            + x_i[0] * x_j1[1] * x_j3[2]
            + x_i[1] * x_j2[0] * x_j3[2]
            - x_j1[1] * x_j2[0] * x_j3[2]
            - x_i[0] * x_j2[1] * x_j3[2]
            + x_j1[0] * x_j2[1] * x_j3[2]
    )


class DynamicsFactory3D(AbstractDynamicsFactory):

    def get_edges_features_matrix(self, elements, nodes):
        return get_edges_features_matrix_numba(elements, nodes)

    @property
    def dimension(self) -> int:
        return DIMENSION

    def calculate_constitutive_matrices(self,
                                        W, mu, lambda_
                                        ):
        X11 = (2 * mu + lambda_) * W[0, 0] + mu * W[1, 1] + lambda_ * W[2, 2]
        X22 = mu * W[0, 0] + (2 * mu + lambda_) * W[1, 1] + lambda_ * W[2, 2]
        X33 = mu * W[0, 0] + lambda_ * W[1, 1] + (2 * mu + lambda_) * W[2, 2]

        X21 = mu * W[1, 0] + lambda_ * W[0, 1]
        X31 = mu * W[2, 0] + lambda_ * W[0, 2]
        X32 = mu * W[2, 1] + lambda_ * W[1, 2]

        X12 = lambda_ * W[1, 0] + mu * W[0, 1]
        X13 = lambda_ * W[2, 0] + mu * W[0, 2]
        X23 = lambda_ * W[2, 1] + mu * W[1, 2]

        return np.block([[X11, X12, X13], [X21, X22, X23], [X31, X32, X33]])

    def calculate_acceleration(self, U, density):
        Z = np.zeros_like(U)
        return density * np.block([[U, Z, Z], [Z, U, Z], [Z, Z, U]])

    def calculate_temperature_C(self, V, C_coef):
        Z = np.zeros_like(V[0])
        X11 = C_coef[0][0] * V[0] + C_coef[0][1] * V[1] + C_coef[0][2] * V[2]
        X22 = C_coef[1][0] * V[0] + C_coef[1][1] * V[1] + C_coef[1][2] * V[2]
        X33 = C_coef[2][0] * V[0] + C_coef[2][1] * V[1] + C_coef[2][2] * V[2]
        return np.block([[X11, Z, Z], [Z, X22, Z], [Z, Z, X33]])

    def calculate_temperature_K(self, W, K_coef):
        return (
                K_coef[0][0] * W[0, 0]
                + K_coef[0][1] * W[0, 1]
                + K_coef[0][2] * W[0, 2]
                + K_coef[1][0] * W[1, 0]
                + K_coef[1][1] * W[1, 1]
                + K_coef[1][2] * W[1, 2]
                + K_coef[2][0] * W[2, 0]
                + K_coef[2][1] * W[2, 1]
                + K_coef[2][2] * W[2, 2]
        )
