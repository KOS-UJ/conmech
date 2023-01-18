import numba
import numpy as np
import scipy.sparse

from conmech.dynamics.factory._abstract_dynamics_factory import (
    AbstractDynamicsFactory,
    get_coo_sparse_data_numba,
)

DIMENSION = 2
ELEMENT_NODES_COUNT = 3
CONNECTED_EDGES_COUNT = 2
INT_PH = 1 / ELEMENT_NODES_COUNT
U_DIVIDER = 12
FEATURE_MATRIX_COUNT = 2 + DIMENSION + DIMENSION**2
VOLUME_DIVIDER = 2


@numba.njit
def get_edges_features_dictionary_numba(elements, nodes):
    # integral of phi over the element (in 2D: 1/3, in 3D: 1/4)
    elements_count, element_size = elements.shape

    edges_features_dict = {}
    element_initial_volume = np.zeros(elements_count)
    dx_dict = {}

    for element_index in range(elements_count):  # TODO: #65 prange?
        element = elements[element_index]
        element_nodes = nodes[element]

        # TODO: #65 Get rid of repetition (?)
        for i in range(element_size):
            i_integrals = get_integral_parts_numba(element_nodes, i)
            i_d_phi_vec = i_integrals[:DIMENSION]
            element_volume = i_integrals[DIMENSION]
            # TODO: #65 Avoid repetition
            element_initial_volume[element_index] = element_volume
            key = element_index, element[i]
            dx_dict[key] = i_d_phi_vec

            for j in range(element_size):
                j_integrals = get_integral_parts_numba(element_nodes, j)
                j_d_phi_vec = j_integrals[:DIMENSION]

                volume_at_nodes = (i != j) * (INT_PH / CONNECTED_EDGES_COUNT)
                # divide by edge count - info about each triangle is "sent" to node via all
                # connected edges (in 2D: 2, in 3D: 3) and summed (by dot product with matrix)
                u = (1 + (i == j)) / U_DIVIDER
                # in 3D: divide by 10 or 20, in 2D: divide by 6 or 12

                v = [INT_PH * j_d_phi for j_d_phi in j_d_phi_vec]

                w = [[i_d_phi * j_d_phi for j_d_phi in j_d_phi_vec] for i_d_phi in i_d_phi_vec]

                result = element_volume * np.array(
                    [volume_at_nodes, u, v[0], v[1], w[0][0], w[0][1], w[1][0], w[1][1]]
                )

                key = element[i], element[j]
                if key in edges_features_dict:
                    edges_features_dict[key] += result
                else:
                    edges_features_dict[key] = result

    return edges_features_dict, element_initial_volume, dx_dict


@numba.njit
def get_edges_features_matrix_numba(elements, nodes):
    # integral of phi over the element (in 2D: 1/3, in 3D: 1/4)
    nodes_count = len(nodes)
    elements_count, element_size = elements.shape

    edges_features_matrix = np.zeros(
        (FEATURE_MATRIX_COUNT, nodes_count, nodes_count), dtype=np.double
    )
    element_initial_volume = np.zeros(elements_count)

    for element_index in range(elements_count):  # TODO: #65 prange?
        element = elements[element_index]
        element_nodes = nodes[element]

        # TODO: #65 Get rid of repetition (?)
        for i in range(element_size):
            i_integrals = get_integral_parts_numba(element_nodes, i)
            i_d_phi_vec = i_integrals[:DIMENSION]
            element_volume = i_integrals[DIMENSION]
            # TODO: #65 Avoid repetition
            element_initial_volume[element_index] = element_volume

            for j in range(element_size):
                j_integrals = get_integral_parts_numba(element_nodes, j)
                j_d_phi_vec = j_integrals[:DIMENSION]

                volume_at_nodes = (i != j) * (INT_PH / CONNECTED_EDGES_COUNT)
                # divide by edge count - info about each triangle is "sent" to node via all
                # connected edges (in 2D: 2, in 3D: 3) and summed (by dot product with matrix)
                u = (1 + (i == j)) / U_DIVIDER
                # in 3D: divide by 10 or 20, in 2D: divide by 6 or 12

                v = [INT_PH * j_d_phi for j_d_phi in j_d_phi_vec]

                w = [[i_d_phi * j_d_phi for j_d_phi in j_d_phi_vec] for i_d_phi in i_d_phi_vec]

                edges_features_matrix[:, element[i], element[j]] += element_volume * np.array(
                    [volume_at_nodes, u, v[0], v[1], w[0][0], w[0][1], w[1][0], w[1][1]]
                )

    return edges_features_matrix, element_initial_volume


@numba.njit
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


@numba.njit
def shoelace_area_numba(nodes):
    x = nodes[:, 0].copy()
    y = nodes[:, 1].copy()
    area = 0.5 * np.abs(np.dot(x, np.roll(y, 1)) - np.dot(y, np.roll(x, 1)))
    return area


@numba.njit
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
    def get_edges_features_dictionary(self, elements, nodes):
        return get_edges_features_dictionary_numba(elements, nodes)

    @property
    def dimension(self) -> int:
        return DIMENSION

    def to_dx_matrix(self, dx_dict: dict, elements_count: int, nodes_count: int):
        _ = self
        keys = np.array(list(dx_dict.keys()), dtype=np.int64)
        values = np.array(list(dx_dict.values()), dtype=np.float64)
        row, col, data = get_coo_sparse_data_numba(keys=keys, values=values)
        shape = (nodes_count, elements_count)

        dx_x = scipy.sparse.coo_matrix((data[0], (row, col)), shape=shape)
        dx_y = scipy.sparse.coo_matrix((data[1], (row, col)), shape=shape)

        dx = scipy.sparse.bmat([[dx_x], [dx_y]], format="csr")
        return dx

    def calculate_constitutive_matrices(self, W, mu, lambda_):
        A_11 = (2 * mu + lambda_) * W[0, 0] + mu * W[1, 1]
        A_12 = mu * W[1, 0] + lambda_ * W[0, 1]
        A_21 = lambda_ * W[1, 0] + mu * W[0, 1]
        A_22 = mu * W[0, 0] + (2 * mu + lambda_) * W[1, 1]
        return scipy.sparse.bmat([[A_11, A_12], [A_21, A_22]], format="csr")

    def calculate_acceleration(self, U, density):
        Z = scipy.sparse.csr_matrix(U.shape)
        return density * scipy.sparse.bmat([[U, Z], [Z, U]], format="csr")

    def calculate_thermal_expansion(self, V, coeff):
        A_11 = coeff[0][0] * V[0] + coeff[0][1] * V[1]
        A_22 = coeff[1][0] * V[0] + coeff[1][1] * V[1]
        return scipy.sparse.bmat([[A_11, A_22]], format="csr")

    def calculate_thermal_conductivity(self, W, coeff):
        return scipy.sparse.csr_matrix(
            coeff[0][0] * W[0, 0]
            + coeff[0][1] * W[0, 1]
            + coeff[1][0] * W[1, 0]
            + coeff[1][1] * W[1, 1]
        )

    def get_piezoelectric_tensor(self, W, coeff):
        A_11 = coeff[0][0][0] * W[0, 0] + coeff[0][1][1] * W[1, 1]
        A_12 = coeff[0][1][0] * W[1, 0] + coeff[0][0][1] * W[0, 1]
        A_21 = coeff[1][1][0] * W[1, 0] + coeff[1][0][1] * W[0, 1]
        A_22 = coeff[1][0][0] * W[0, 0] + coeff[1][1][1] * W[1, 1]
        return scipy.sparse.bmat([[A_11 + A_12, A_22 + A_21]], format="csr")

    def get_permittivity_tensor(self, W, coeff):
        return (
            coeff[0][0] * W[0, 0]
            + coeff[0][1] * W[0, 1]
            + coeff[1][0] * W[1, 0]
            + coeff[1][1] * W[1, 1]
        )
