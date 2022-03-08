import numba
import numpy as np
from numba import njit


@njit  # (parallel=True)
def get_edges_features_matrix_numba(elements, nodes):
    ELEMENT_NODES_COUNT = 4
    CONNECTED_EDGES_COUNT = 3
    int_Ph = 1 / ELEMENT_NODES_COUNT
    # mean value of of phi over the element (in 2D: 1/3, in 3D: 1/4)
    nodes_count = len(nodes)
    elements_count, element_size = elements.shape

    edges_features_matrix = np.zeros((14, nodes_count, nodes_count), dtype=np.double)
    element_initial_volume = np.zeros(elements_count)

    for element_index in range(elements_count):  # TODO: prange?
        element = elements[element_index]
        element_points = nodes[element]

        # TODO: Get rid of repetition (?)
        for i in range(element_size):
            i_dPhX, i_dPhY, i_dPhZ, element_volume = get_integral_parts_numba(
                element_points, i
            )
            # TODO: Avoid repetition
            element_initial_volume[element_index] = element_volume

            for j in range(element_size):
                j_dPhX, j_dPhY, j_dPhZ, _ = get_integral_parts_numba(element_points, j)

                volume = (i != j) * (int_Ph / CONNECTED_EDGES_COUNT)
                # divide by edge count - info about each triangle is "sent" to node via all connected edges
                # (in 2D: 2, in 3D: 3) and summed (by dot product with matrix)
                u = (1 + (i == j)) / 20.0
                # in 3D: divide by 10 or 20, in 2D: divide by 6 or 12

                v1 = int_Ph * j_dPhX
                v2 = int_Ph * j_dPhY
                v3 = int_Ph * j_dPhZ

                w11 = i_dPhX * j_dPhX
                w12 = i_dPhX * j_dPhY
                w13 = i_dPhX * j_dPhZ

                w21 = i_dPhY * j_dPhX
                w22 = i_dPhY * j_dPhY
                w23 = i_dPhY * j_dPhZ

                w31 = i_dPhZ * j_dPhX
                w32 = i_dPhZ * j_dPhY
                w33 = i_dPhZ * j_dPhZ

                edges_features_matrix[:, element[i], element[j]] += (
                    element_volume
                    * np.array(
                        [
                            volume,
                            u,
                            v1,
                            v2,
                            v3,
                            w11,
                            w12,
                            w13,
                            w21,
                            w22,
                            w23,
                            w31,
                            w32,
                            w33,
                        ]
                    )
                )

    return edges_features_matrix, element_initial_volume


@njit
def get_integral_parts_numba(element_nodes, element_index):
    ELEMENT_NODES_COUNT = 4
    x_i = element_nodes[element_index]
    x_j1, x_j2, x_j3 = list(
        element_nodes[np.arange(ELEMENT_NODES_COUNT) != element_index]
    )

    dm = denominator_numba(x_i, x_j1, x_j2, x_j3)
    element_volume = np.abs(dm) / 6.0

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


######################################


def calculate_constitutive_matrices(
    W11, W12, W13, W21, W22, W23, W31, W32, W33, MU, LA
):
    X11 = (2 * MU + LA) * W11 + MU * W22 + LA * W33
    X22 = MU * W11 + (2 * MU + LA) * W22 + LA * W33
    X33 = MU * W11 + LA * W22 + (2 * MU + LA) * W33

    X21 = MU * W21 + LA * W12
    X31 = MU * W31 + LA * W13
    X32 = MU * W32 + LA * W23

    X12 = LA * W21 + MU * W12
    X13 = LA * W31 + MU * W13
    X23 = LA * W32 + MU * W23

    return np.block([[X11, X12, X13], [X21, X22, X23], [X31, X32, X33]])


def create_acceleration(U, density):
    Z = np.zeros_like(U)
    return density * np.block([[U, Z, Z], [Z, U, Z], [Z, Z, U]])


def get_matrices(edges_features_matrix, MU, LA, TH, ZE, density, time_step, slice_ind):
    VOL = edges_features_matrix[0]
    U = edges_features_matrix[1][slice_ind, slice_ind]

    ALL_V = [edges_features_matrix[i][slice_ind, slice_ind] for i in range(2, 5)]
    ALL_W = [edges_features_matrix[i][slice_ind, slice_ind] for i in range(5, 14)]

    A = calculate_constitutive_matrices(*ALL_W, TH, ZE)
    B = calculate_constitutive_matrices(*ALL_W, MU, LA)
    ACC = create_acceleration(U, density)

    A_plus_B_times_ts = A + B * time_step
    C = ACC + A_plus_B_times_ts * time_step

    K = None
    C2X = None
    C2Y = None
    
    return C, B, VOL, A_plus_B_times_ts, A, ACC, K, C2X, C2Y

