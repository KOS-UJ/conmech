import numpy as np
from numba import njit

from deep_conmech.simulator.dynamics.factory._abstract_dynamics_factory import \
    AbstractDynamicsFactory


DIMENSION = 2
ELEMENT_NODES_COUNT = 3
CONNECTED_EDGES_COUNT = 2
INT_PH = 1 / ELEMENT_NODES_COUNT
U_DIVIDER = 12
FEATURE_MATRIX_COUNT = 2 + DIMENSION + DIMENSION ** 2
VOLUME_DIVIDER = 2


@njit
def get_edges_features_matrix_numba(elements, nodes):
    # integral of phi over the element (in 2D: 1/3, in 3D: 1/4)
    nodes_count = len(nodes)
    elements_count, element_size = elements.shape

    edges_features_matrix = np.zeros((FEATURE_MATRIX_COUNT, nodes_count, nodes_count), dtype=np.double)
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
                u = (1 + (i == j)) / U_DIVIDER
                # in 3D: divide by 10 or 20, in 2D: divide by 6 or 12

                v1 = INT_PH * j_dPhX
                v2 = INT_PH * j_dPhY

                w11 = i_dPhX * j_dPhX
                w12 = i_dPhX * j_dPhY
                w21 = i_dPhY * j_dPhX
                w22 = i_dPhY * j_dPhY

                edges_features_matrix[:, element[i], element[j]] += element_volume * np.array([
                    volume,
                    u,
                    v1, v2,
                    w11, w12,
                    w21, w22
                ])

    return edges_features_matrix, element_initial_volume


@njit
def get_integral_parts_numba(element_nodes, element_index):
    x_i = element_nodes[element_index]
    x_j1, x_j2 = list(element_nodes[np.arange(ELEMENT_NODES_COUNT) != element_index])

    dm = denominator_numba(x_i, x_j1, x_j2)
    element_volume = np.abs(dm) / VOLUME_DIVIDER

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


class DynamicsFactory2D(AbstractDynamicsFactory):
    def get_edges_features_matrix(self, elements, nodes):
        return get_edges_features_matrix_numba(elements, nodes)

    @property
    def dimension(self) -> int:
        return DIMENSION

    def calculate_constitutive_matrices(self, W, MU, LA):
        X11 = (2 * MU + LA) * W[0, 0] + MU * W[1, 1]
        X22 = MU * W[0, 0] + (2 * MU + LA) * W[1, 1]
        X12 = MU * W[1, 0] + LA * W[0, 1]
        X21 = LA * W[1, 0] + MU * W[0, 1]
        return np.block([[X11, X12], [X21, X22]])

    def calculate_acceleration(self, U, density):
        Z = np.zeros_like(U)
        return density * np.block([[U, Z], [Z, U]])

    def calculate_temperature_C(self, V, C_coef):
        Z = np.zeros_like(V[1])
        X11 = C_coef[0][0] * V[0] + C_coef[0][1] * V[1]
        X22 = C_coef[1][0] * V[0] + C_coef[1][1] * V[1]
        return np.block([[X11, Z], [Z, X22]])

    def calculate_temperature_K(self, W, K_coef):
        return (
                K_coef[0][0] * W[0, 0]
                + K_coef[0][1] * W[0, 1]
                + K_coef[1][0] * W[1, 0]
                + K_coef[1][1] * W[1, 1]
        )
