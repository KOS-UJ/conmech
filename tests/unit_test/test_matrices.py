import numpy as np
import pytest
from conmech.features.boundaries import identify_surfaces, get_boundaries
from deep_conmech.simulator.setting.matrices_3d import get_edges_features_matrix_3d_numba
from deep_conmech.simulator.mesh import mesh_builders
from deep_conmech.simulator.setting.setting_matrices import get_edges_features_matrix_2d_numba


def test_matrices_unit_cube_2d_integrals():
    # Arrange
    initial_nodes, elements = mesh_builders.build_mesh(
        mesh_type="meshzoo_rectangle", mesh_density_x=3, scale_x=1, scale_y=1
    )      
    edges_features_matrix, element_initial_volume = get_edges_features_matrix_2d_numba(
        elements, initial_nodes
    )

    # Act and Assert
    np.testing.assert_allclose(element_initial_volume.sum(),1)

    VOL = edges_features_matrix[0]
    U = edges_features_matrix[1]
    np.testing.assert_allclose(VOL.sum(),1)
    np.testing.assert_allclose(U.sum(),1)
    
    ALL_V = [edges_features_matrix[i] for i in range(2, 1)]
    ALL_W = [edges_features_matrix[i] for i in range(5, 8)]

    for M in (*ALL_V, *ALL_W):
        np.testing.assert_almost_equal(M.sum(), 0)

    #TODO: Check Wij = Wji.T



def test_matrices_unit_cube_3d_integrals():
    # Arrange
    initial_nodes, elements = mesh_builders.build_mesh(
        mesh_type="meshzoo_cube_3d", mesh_density_x=3
    )      
    edges_features_matrix, element_initial_volume = get_edges_features_matrix_3d_numba(
        elements, initial_nodes
    )

    # Act and Assert
    #np.testing.assert_allclose(element_initial_volume.sum(),1)

    VOL = edges_features_matrix[0]
    U = edges_features_matrix[1]
    np.testing.assert_allclose(VOL.sum(),1)
    np.testing.assert_allclose(U.sum(),1)
    
    ALL_V = [edges_features_matrix[i] for i in range(2, 5)]
    ALL_W = [edges_features_matrix[i] for i in range(5, 14)]

    for M in (*ALL_V, *ALL_W):
        np.testing.assert_almost_equal(M.sum(), 0)

    #TODO: Check Wij = Wji.T


