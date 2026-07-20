# CONMECH @ Jagiellonian University in Kraków
#
# Copyright (C) 2022-2026  Piotr Bartman-Szwarc <piotr.bartman@uj.edu.pl>
# Copyright (C) 2022  Michał Jureczka <michal.jureczka@uj.edu.pl>
#
# This program is free software; you can redistribute it and/or
# modify it under the terms of the GNU General Public License
# as published by the Free Software Foundation; either version 3
# of the License, or (at your option) any later version.
#
# This program is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
# GNU General Public License for more details.
#
# You should have received a copy of the GNU General Public License
# along with this program; if not, write to the Free Software
# Foundation, Inc., 51 Franklin Street, Fifth Floor, Boston, MA  02110-1301,
# USA.
# pylint: disable=R0914
import numba
import numpy as np

from conmech.dynamics.factory._abstract_dynamics_factory import AbstractDynamicsFactory
from conmech.helpers.assembly import block, hstack_blocks, coo_features_to_csr
from conmech.struct.stiffness_matrix import SM2, SM1, SM1to2

DIMENSION = 2
ELEMENT_NODES_COUNT = 3
CONNECTED_EDGES_COUNT = 2
INT_PH = 1 / ELEMENT_NODES_COUNT
U_DIVIDER = 12
FEATURE_MATRIX_COUNT = 3 + DIMENSION + DIMENSION**2
VOLUME_DIVIDER = 2


@numba.njit
def get_edges_features_matrix_coo_numba(elements, nodes):
    """Assemble node features in COO form.

    Contains integral of phi over the element (in 2D: 1/3, in 3D: 1/4).
    Returns the COO triplets ``(rows, cols, data)`` with ``data`` of shape
    ``(FEATURE_MATRIX_COUNT, nnz)`` where ``nnz = elements_count * element_size**2``.
    Duplicate ``(row, col)`` entries are *not* summed here - that is left to
    ``scipy.sparse`` (which sums duplicates on construction).
    """
    elements_count, element_size = elements.shape
    nnz = elements_count * element_size * element_size

    rows = np.empty(nnz, dtype=np.int64)
    cols = np.empty(nnz, dtype=np.int64)
    data = np.zeros((FEATURE_MATRIX_COUNT, nnz), dtype=np.double)

    element_initial_volume = np.zeros(elements_count)
    # Local stifness matrices (w[0, 0], w[0, 1], w[1, 0], w[1, 1]) per mesh element
    # Detailed description can be found in [LSM] Local stifness matrix
    local_stifness_matrices = np.empty(
        (DIMENSION, DIMENSION, elements_count, element_size, element_size)
    )

    en0 = np.empty(ELEMENT_NODES_COUNT)
    en1 = np.empty(ELEMENT_NODES_COUNT)
    entry = 0
    for element_index in range(elements_count):
        element = elements[element_index]
        element_nodes = nodes[element]

        int_sqr = (
            np.sum(element_nodes**2, axis=0)
            + element_nodes[0] * (element_nodes[1] + element_nodes[2])
            + element_nodes[1] * element_nodes[2]
        )
        en0[:] = element_nodes[:, 0]
        en1[:] = element_nodes[:, 1]
        int_xy = en0 @ np.array([[2.0, 1.0, 1.0], [1.0, 2.0, 1.0], [1.0, 1.0, 2.0]]) @ en1
        int_matrix = np.array(
            [
                [
                    0,
                    1 / 6 * en0.sum(),
                    1 / 6 * en1.sum(),
                ],
                [
                    1 / 6 * en0.sum(),
                    1 / 12 * int_sqr[0],
                    1 / 24 * int_xy,
                ],
                [
                    1 / 6 * en1.sum(),
                    1 / 24 * int_xy,
                    1 / 12 * int_sqr[1],
                ],
            ]
        )
        jacobian = np.abs(np.linalg.det(element_nodes[1:] - element_nodes[0]))

        # TODO: #65 Get rid of repetition (?)
        for i in range(element_size):
            i_integrals = get_integral_parts_numba(element_nodes, i)
            i_d_phi_vec = i_integrals[:DIMENSION]
            element_volume = i_integrals[DIMENSION]
            # TODO: #65 Avoid repetition
            element_initial_volume[element_index] = element_volume
            vert_ip1 = element_nodes[(i + 1) % 3]
            vert_ip2 = element_nodes[(i + 2) % 3]
            c_i = np.linalg.det(np.row_stack((vert_ip1, vert_ip2)))
            d_i = vert_ip1[1] - vert_ip2[1]
            e_i = vert_ip2[0] - vert_ip1[0]
            coeffs_i = np.array([c_i, d_i, e_i])

            for j in range(element_size):
                vert_jp1 = element_nodes[(j + 1) % 3]
                vert_jp2 = element_nodes[(j + 2) % 3]
                c_j = np.linalg.det(np.row_stack((vert_jp1, vert_jp2)))
                d_j = vert_jp1[1] - vert_jp2[1]
                e_j = vert_jp2[0] - vert_jp1[0]
                coeffs_j = np.array([c_j, d_j, e_j])
                q = jacobian * coeffs_i @ int_matrix @ coeffs_j / (
                    4 * element_volume**2
                ) + c_i * c_j / (4 * element_volume)

                j_integrals = get_integral_parts_numba(element_nodes, j)
                j_d_phi_vec = j_integrals[:DIMENSION]

                volume_at_nodes = (i != j) * (INT_PH / CONNECTED_EDGES_COUNT)
                # divide by edge count - info about each triangle is "sent" to node via all
                # connected edges (in 2D: 2, in 3D: 3) and summed (by dot product with matrix)
                u = (1 + (i == j)) / U_DIVIDER
                # in 3D: divide by 10 or 20, in 2D: divide by 6 or 12

                v = [INT_PH * j_d_phi for j_d_phi in j_d_phi_vec]

                w = [[i_d_phi * j_d_phi for j_d_phi in j_d_phi_vec] for i_d_phi in i_d_phi_vec]

                local_stifness_matrices[:, :, element_index, i, j] = element_volume * np.asarray(w)

                rows[entry] = element[i]
                cols[entry] = element[j]
                data[:, entry] = element_volume * np.array(
                    [
                        volume_at_nodes,
                        u,
                        v[0],
                        v[1],
                        w[0][0],
                        w[0][1],
                        w[1][0],
                        w[1][1],
                        q / element_volume,
                    ]
                )
                entry += 1

    return rows, cols, data, element_initial_volume, local_stifness_matrices


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
        rows, cols, data, element_initial_volume, local_stifness_matrices = (
            get_edges_features_matrix_coo_numba(elements, nodes)
        )
        features = coo_features_to_csr(rows, cols, data, len(nodes))
        return features, element_initial_volume, local_stifness_matrices

    @property
    def dimension(self) -> int:
        return DIMENSION

    def calculate_constitutive_matrices(self, W: np.ndarray, mu: float, lambda_: float) -> SM2:
        A_11 = (2 * mu + lambda_) * W[0, 0] + mu * W[1, 1]
        A_12 = mu * W[1, 0] + lambda_ * W[0, 1]
        A_21 = lambda_ * W[1, 0] + mu * W[0, 1]
        A_22 = mu * W[0, 0] + (2 * mu + lambda_) * W[1, 1]
        return SM2(block([[A_11, A_12], [A_21, A_22]]))

    def get_relaxation_tensor(self, W, coeff):
        A_11 = coeff[0][0][0] * W[0, 0] + coeff[0][1][1] * W[1, 1]
        A_12 = coeff[0][1][0] * W[1, 0] + coeff[0][0][1] * W[0, 1]
        A_21 = coeff[1][1][0] * W[1, 0] + coeff[1][0][1] * W[0, 1]
        A_22 = coeff[1][0][0] * W[0, 0] + coeff[1][1][1] * W[1, 1]
        return SM2(block([[A_11, A_12], [A_21, A_22]]))

    def calculate_acceleration(self, U, density):
        return SM2(density * block([[U, None], [None, U]]))

    def calculate_thermal_expansion(self, V, coeff):
        A_11 = coeff[0][0] * V[0] + coeff[0][1] * V[1]
        A_22 = coeff[1][0] * V[0] + coeff[1][1] * V[1]
        return SM1to2(hstack_blocks([A_11, A_22]))

    def calculate_thermal_conductivity(self, W, coeff):
        return SM1(
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
        return SM2(hstack_blocks([A_11 + A_12, A_22 + A_21]))

    def get_permittivity_tensor(self, W, coeff):
        return SM1(
            coeff[0][0] * W[0, 0]
            + coeff[0][1] * W[0, 1]
            + coeff[1][0] * W[1, 0]
            + coeff[1][1] * W[1, 1]
        )
