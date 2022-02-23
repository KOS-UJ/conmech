import numpy as np
import numba
from numba import njit



#@njit  # (parallel=True)
def get_edges_features_matrix_numba(elements, nodes):
    nodes_count = len(nodes)
    elements_count, element_size = elements.shape
    dim=element_size-1

    edges_features_matrix = np.zeros((nodes_count, nodes_count, 11), dtype=np.double)
    element_initial_volume = np.zeros(elements_count)

    for element_index in range(elements_count):  # TODO: prange?
        element = elements[element_index]
        element_points = nodes[element]

        # TODO: Get rid of repetition (?)
        for i in range(element_size):
            i_dPhX, i_dPhY, i_dPhZ, element_volume = get_integral_parts_numba(element_points, i)
            # TODO: Avoid repetition
            element_initial_volume[element_index] = element_volume

            for j in range(element_size):
                j_dPhX, j_dPhY, j_dPhZ, _ = get_integral_parts_numba(element_points, j)

                volume = (i != j) / (6.0 * 4.0) # in 2D: divide by 3 in for element integral | divide by 2 for each edge (3D: 4 - each face)
                u = (1 + (i == j)) / 20.0 # in 2D: divide by 6 or 12
                #u1 = i_dPhX / 3.0
                #u2 = i_dPhY / 3.0
                #u3 = i_dPhZ / 3.0

                w11 = i_dPhX * j_dPhX
                w12 = i_dPhX * j_dPhY
                w13 = i_dPhX * j_dPhZ

                w21 = i_dPhY * j_dPhX
                w22 = i_dPhY * j_dPhY
                w23 = i_dPhY * j_dPhZ

                w31 = i_dPhZ * j_dPhX
                w32 = i_dPhZ * j_dPhY
                w33 = i_dPhZ * j_dPhZ

                edges_features_matrix[element[i], element[j]] += element_volume * np.array(
                    [volume, w11, w12, w13, w21, w22, w23, w31, w32, w33, u]
                )

    return edges_features_matrix, element_initial_volume


#@njit
def get_integral_parts_numba(element_nodes, element_index):
    x_i = element_nodes[element_index % 4]
    x_j1 = element_nodes[(element_index + 1) % 4]
    x_j2 = element_nodes[(element_index + 2) % 4]
    x_j3 = element_nodes[(element_index + 3) % 4]

    dm = denominator_numba(x_i, x_j1, x_j2, x_j3)
    element_volume = np.abs(dm) / 6.0

    x_sub = x_j1[2]*x_j2[1] - x_j1[1]*x_j2[2] - x_j1[2]*x_j3[1] + x_j2[2]*x_j3[1] + x_j1[1]*x_j3[2] - x_j2[1]*x_j3[2]
    y_sub = x_j1[0]*x_j2[2] - x_j1[2]*x_j2[0] + x_j1[2]*x_j3[0] - x_j2[2]*x_j3[0] - x_j1[0]*x_j3[2] + x_j2[0]*x_j3[2]
    z_sub = x_j1[1]*x_j2[0] - x_j1[0]*x_j2[1] - x_j1[1]*x_j3[0] + x_j2[1]*x_j3[0] + x_j1[0]*x_j3[1] - x_j2[0]*x_j3[1]

    dPhX = y_sub / dm
    dPhY = x_sub / dm
    dPhZ = z_sub / dm

    return dPhX, dPhY, dPhZ, element_volume


@njit
def denominator_numba(x_i, x_j1, x_j2, x_j3):
    return (
    x_i[2]*x_j1[1]*x_j2[0] - x_i[1]*x_j1[2]*x_j2[0] - x_i[2]*x_j1[0]*x_j2[1] + x_i[0]*x_j1[2]*x_j2[1] + x_i[1]*x_j1[0]*x_j2[2]
    - x_i[0]*x_j1[1]*x_j2[2] - x_i[2]*x_j1[1]*x_j3[0] + x_i[1]*x_j1[2]*x_j3[0] + x_i[2]*x_j2[1]*x_j3[0] - x_j1[2]*x_j2[1]*x_j3[0]
    - x_i[1]*x_j2[2]*x_j3[0] + x_j1[1]*x_j2[2]*x_j3[0] + x_i[2]*x_j1[0]*x_j3[1] - x_i[0]*x_j1[2]*x_j3[1] - x_i[2]*x_j2[0]*x_j3[1]
    + x_j1[2]*x_j2[0]*x_j3[1] + x_i[0]*x_j2[2]*x_j3[1] - x_j1[0]*x_j2[2]*x_j3[1] - x_i[1]*x_j1[0]*x_j3[2] + x_i[0]*x_j1[1]*x_j3[2]
    + x_i[1]*x_j2[0]*x_j3[2] - x_j1[1]*x_j2[0]*x_j3[2] - x_i[0]*x_j2[1]*x_j3[2] + x_j1[0]*x_j2[1]*x_j3[2]
    )
