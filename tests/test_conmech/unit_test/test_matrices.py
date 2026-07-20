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
import numpy as np

from conmech.dynamics.factory._dynamics_factory_2d import (
    get_edges_features_matrix_coo_numba as sut_2d,
)
from conmech.dynamics.factory._dynamics_factory_3d import (
    get_edges_features_matrix_coo_numba as sut_3d,
)
from conmech.helpers.assembly import coo_features_to_csr
from conmech.dynamics.dynamics import Dynamics
from conmech.simulations.problem_solver import Body
from conmech.mesh.mesh import Mesh
from conmech.mesh import mesh_builders
from conmech.properties.mesh_description import (
    RectangleMeshDescription,
    CubeMeshDescription,
)


def _assemble_2d(nodes, elements):
    rows, cols, data, element_initial_volume, local_stiff = sut_2d(elements=elements, nodes=nodes)
    features = coo_features_to_csr(rows, cols, data, len(nodes))
    return features, element_initial_volume, local_stiff


def _assemble_3d(nodes, elements):
    rows, cols, data, element_initial_volume, local_stiff = sut_3d(elements=elements, nodes=nodes)
    features = coo_features_to_csr(rows, cols, data, len(nodes))
    return features, element_initial_volume, local_stiff


def test_matrices_2d_integrals():
    # Arrange
    scale_x = 2
    scale_y = 3
    area = scale_x * scale_y
    initial_nodes, elements = mesh_builders.build_mesh(
        mesh_descr=RectangleMeshDescription(
            initial_position=None,
            max_element_perimeter=(scale_x / 3),
            scale=[scale_x, scale_y],
        )
    )

    # Act
    edges_features_matrix, element_initial_volume, _ = _assemble_2d(initial_nodes, elements)

    # Assert
    np.testing.assert_allclose(element_initial_volume.sum(), area)

    VOL = edges_features_matrix[0]
    U = edges_features_matrix[1]
    np.testing.assert_allclose(VOL.sum(), area)
    np.testing.assert_allclose(U.sum(), area)

    ALL_V = [edges_features_matrix[i] for i in range(2, 4)]
    ALL_W = [edges_features_matrix[i] for i in range(4, 8)]

    for M in (*ALL_V, *ALL_W):
        np.testing.assert_almost_equal(M.sum(), 0)

    # Memory: nnz stored is elements_count * element_size**2, far below N**2.
    nnz = elements.shape[0] * elements.shape[1] ** 2
    assert nnz < len(initial_nodes) ** 2
    assert U.nnz <= nnz


def test_matrices_3d_integrals():
    # Arrange
    initial_nodes, elements = mesh_builders.build_mesh(
        mesh_descr=CubeMeshDescription(initial_position=None)
    )

    # Act
    edges_features_matrix, element_initial_volume, _ = _assemble_3d(initial_nodes, elements)

    # Assert
    np.testing.assert_allclose(element_initial_volume.sum(), 1)

    VOL = edges_features_matrix[0]
    U = edges_features_matrix[1]
    np.testing.assert_allclose(VOL.sum(), 1)
    np.testing.assert_allclose(U.sum(), 1)

    ALL_V = [edges_features_matrix[i] for i in range(2, 5)]
    ALL_W = [edges_features_matrix[i] for i in range(5, 14)]

    for M in (*ALL_V, *ALL_W):
        np.testing.assert_almost_equal(M.sum(), 0)

    nnz = elements.shape[0] * elements.shape[1] ** 2
    assert nnz < len(initial_nodes) ** 2


def test_local_stiff_mats_assembly():
    # Arrange
    dimension = 2
    scale_x = 2
    scale_y = 3
    initial_nodes, elements = mesh_builders.build_mesh(
        mesh_descr=RectangleMeshDescription(
            initial_position=None,
            max_element_perimeter=(scale_x / 3),
            scale=[scale_x, scale_y],
        )
    )
    edges_features_matrix, _, local_stiff_mats = _assemble_2d(initial_nodes, elements)
    from conmech.struct.types import FeatureMatrix

    expected_w_matrix = FeatureMatrix(
        [
            [edges_features_matrix[2 + dimension * (k + 1) + j] for j in range(dimension)]
            for k in range(dimension)
        ]
    )

    mesh = object.__new__(Mesh)
    mesh.nodes = initial_nodes
    mesh.elements = elements

    body = object.__new__(Body)
    body.mesh = mesh

    dynamics = object.__new__(Dynamics)
    dynamics.body = body
    dynamics._local_stifness_matrices = local_stiff_mats
    dynamics._w_matrix = expected_w_matrix
    density = np.ones(elements.shape[0])

    # Act
    assembled_w_mat = dynamics.asembly_w_matrix_with_density(density)

    # Assert - identical blocks (density == 1 reproduces the plain W operator)
    for k in range(dimension):
        for m in range(dimension):
            np.testing.assert_almost_equal(
                assembled_w_mat[k, m].toarray(), expected_w_matrix[k, m].toarray()
            )
