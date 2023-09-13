import numpy as np

from conmech.dynamics.factory._dynamics_factory_2d import (
    get_edges_features_matrix_numba as sut_2d,
)
from conmech.dynamics.factory._dynamics_factory_3d import (
    get_edges_features_matrix_numba as sut_3d,
)
from conmech.dynamics.dynamics import Dynamics
from conmech.simulations.problem_solver import Body
from conmech.mesh.mesh import Mesh
from conmech.mesh import mesh_builders
from conmech.properties.mesh_description import RectangleMeshDescription, CubeMeshDescription


def test_matrices_2d_integrals():
    # Arrange
    scale_x = 2
    scale_y = 3
    area = scale_x * scale_y
    initial_nodes, elements = mesh_builders.build_mesh(
        mesh_descr=RectangleMeshDescription(
            initial_position=None, max_element_perimeter=(scale_x / 3), scale=[scale_x, scale_y]
        )
    )

    # Act
    edges_features_matrix, element_initial_volume, _ = sut_2d(
        elements=elements, nodes=initial_nodes
    )

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

    # TODO: Check Wij = Wji.T


def test_matrices_3d_integrals():
    # Arrange
    initial_nodes, elements = mesh_builders.build_mesh(
        mesh_descr=CubeMeshDescription(initial_position=None)
    )

    # Act
    edges_features_matrix, element_initial_volume, _ = sut_3d(
        elements=elements, nodes=initial_nodes
    )

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

    # TODO: Check Wij = Wji.T


def test_local_stiff_mats_assembly():
    # Arrange
    dimension = 2
    scale_x = 2
    scale_y = 3
    initial_nodes, elements = mesh_builders.build_mesh(
        mesh_descr=RectangleMeshDescription(
            initial_position=None, max_element_perimeter=(scale_x / 3), scale=[scale_x, scale_y]
        )
    )
    edges_features_matrix, _, local_stiff_mats = sut_2d(elements=elements, nodes=initial_nodes)
    expected_w_matrix = np.asarray(
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

    # Assert
    np.testing.assert_almost_equal(assembled_w_mat, expected_w_matrix)
