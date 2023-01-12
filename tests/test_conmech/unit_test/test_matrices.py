import numpy as np

from conmech.dynamics.factory._dynamics_factory_2d import (
    get_edges_features_matrix_numba as sut_2d,
)
from conmech.dynamics.factory._dynamics_factory_3d import (
    get_edges_features_matrix_numba as sut_3d,
)
from conmech.mesh import mesh_builders
from conmech.properties.mesh_properties import MeshProperties


def test_matrices_2d_integrals():
    # Arrange
    scale_x = 2
    scale_y = 3
    area = scale_x * scale_y
    initial_nodes, elements = mesh_builders.build_mesh(
        mesh_prop=MeshProperties(
            dimension=2, mesh_type="meshzoo_rectangle", mesh_density=[3], scale=[scale_x, scale_y]
        ),
    )

    # Act
    edges_features_matrix, element_initial_volume = sut_2d(elements=elements, nodes=initial_nodes)

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
        mesh_prop=MeshProperties(
            dimension=3, mesh_type="meshzoo_cube_3d", mesh_density=[3], scale=[1]
        ),
    )

    # Act
    edges_features_matrix, element_initial_volume = sut_3d(elements=elements, nodes=initial_nodes)

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
