import numpy as np
import numba
from numba import njit

@njit
def get_integral_parts(element_nodes, element_index):
    x_i = element_nodes[element_index % 4]
    x_j1 = element_nodes[(element_index + 1) % 4]
    x_j2 = element_nodes[(element_index + 2) % 4]
    x_j3 = element_nodes[(element_index + 3) % 4]

    dm = denominator(x_i, x_j1, x_j2, x_j2)
    element_volume = np.abs(dm) / 2.0 # = np.abs(dm) / 2.0 = shoelace_area

    x_sub = x_j1[2]*x_j2[1] - x_j1[1]*x_j2[2] - x_j1[2]*x_j3[1] + x_j2[2]*x_j3[1] + x_j1[1]*x_j3[2] - x_j2[1]*x_j3[2]
    y_sub = - x_j1[2]*x_j2[0] + x_j1[0]*x_j2[2] + x_j1[2]*x_j3[0] - x_j2[2]*x_j3[0] - x_j1[0]*x_j3[2] + x_j2[0]*x_j3[2] 
    z_sub = x_j1[1]*x_j2[0] - x_j1[0]*x_j2[1] - x_j1[1]*x_j3[0] + x_j2[1]*x_j3[0] + x_j1[0]*x_j3[1] - x_j2[0]*x_j3[1]

    dPhX = y_sub / dm
    dPhY = x_sub / dm
    dPhZ = z_sub / dm

    return dPhX, dPhY, dPhZ, element_volume


@njit
def denominator(x_i, x_j1, x_j2, x_j3):
    return (
    x_i[2]*x_j1[1]*x_j2[0] - x_i[1]*x_j1[2]*x_j2[0] - x_i[2]*x_j1[0]*x_j2[1] + x_i[0]*x_j1[2]*x_j2[1] + x_i[1]*x_j1[0]*x_j2[2]
    - x_i[0]*x_j1[1]*x_j2[2] - x_i[2]*x_j1[1]*x_j3[0] + x_i[1]*x_j1[2]*x_j3[0] + x_i[2]*x_j2[1]*x_j3[0] - x_j1[2]*x_j2[1]*x_j3[0]
    - x_i[1]*x_j2[2]*x_j3[0] + x_j1[1]*x_j2[2]*x_j3[0] + x_i[2]*x_j1[0]*x_j3[1] - x_i[0]*x_j1[2]*x_j3[1] - x_i[2]*x_j2[0]*x_j3[1]
    + x_j1[2]*x_j2[0]*x_j3[1] + x_i[0]*x_j2[2]*x_j3[1] - x_j1[0]*x_j2[2]*x_j3[1] - x_i[1]*x_j1[0]*x_j3[2] + x_i[0]*x_j1[1]*x_j3[2]
    + x_i[1]*x_j2[0]*x_j3[2] - x_j1[1]*x_j2[0]*x_j3[2] - x_i[0]*x_j2[1]*x_j3[2] + x_j1[0]*x_j2[1]*x_j3[2]
    )
