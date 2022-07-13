import jax.numpy as jnp
import numba
import numpy as np
import scipy.sparse

from conmech.dynamics.factory._abstract_dynamics_factory import AbstractDynamicsFactory
from conmech.helpers import nph

DIMENSION = 3
VOLUME_DIVIDER = 6


@numba.njit
def precomputation_numba(nodes, elements):
    elements_count = len(elements)

    B_m = np.zeros((elements_count, DIMENSION, DIMENSION))
    W = np.zeros(elements_count)
    D_m = np.zeros((elements_count, DIMENSION, DIMENSION))

    for element_index in range(elements_count):  # TODO: #65 prange?
        element = elements[element_index]
        element_nodes = nodes[element]

        fill_matrix_numba(D_m[element_index], element_nodes)

        B_m[element_index] = np.linalg.inv(D_m[element_index])
        W[element_index] = np.abs(np.linalg.det(D_m[element_index]) / VOLUME_DIVIDER)

    return B_m, W, D_m


def fill_matrix(D, element_nodes):
    D[:, 0] = element_nodes[0] - element_nodes[3]
    D[:, 1] = element_nodes[1] - element_nodes[3]
    D[:, 2] = element_nodes[2] - element_nodes[3]


fill_matrix_numba = numba.njit(fill_matrix)


# @numba.njit
# def compute_constitutive_energy_numba_old(nodes, elements, B_m, W):
#     mu = 4
#     lambda_ = 4
#     elements_count = len(elements)

#     D_s = np.zeros((DIMENSION, DIMENSION))

#     energy = 0.0
#     I = np.eye(DIMENSION)
#     for element_index in range(elements_count):  # TODO: #65 prange?
#         element = elements[element_index]
#         element_nodes = nodes[element]

#         fill_matrix_numba(D_s, element_nodes)
#         F = D_s @ B_m[element_index]

#         E = 0.5 * (F.T @ F - I)
#         phi = mu * np.sum(E * E) + 0.5 * lambda_ * np.trace(E) ** 2
#         energy += W[element_index] * phi
#     return energy
