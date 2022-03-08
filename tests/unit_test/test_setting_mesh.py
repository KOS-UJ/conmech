import numpy as np
import pytest
from conmech.helpers import nph
from deep_conmech.simulator.matrices.matrices_2d import (
    get_edges_features_matrix_2d_numba,
)
from deep_conmech.simulator.matrices.matrices_3d import (
    get_edges_features_matrix_3d_numba,
)
from deep_conmech.simulator.mesh import mesh_builders
from deep_conmech.simulator.setting.setting_mesh import SettingMesh


@pytest.mark.parametrize("scale_x, scale_y", ((1, 1), (2, 3), (5, 1)))
def test_boundary_nodes_data_2d(scale_x, scale_y):
    # Arrange
    volume = 2 * (scale_x + scale_y)
    setting = SettingMesh(
        mesh_type="meshzoo_rectangle",
        mesh_density_x=3,
        mesh_density_y=3,
        scale_x=scale_x,
        scale_y=scale_y,
    )
    setting.prepare()

    # Act and Assert
    np.testing.assert_allclose(setting.boundary_nodes_volume.sum(), volume)
    np.testing.assert_allclose(
        nph.euclidean_norm_numba(setting.boundary_normals),
        np.ones((len(setting.boundary_normals), 1)),
    )


def test_boundary_nodes_data_3d():
    # Arrange
    volume = 6
    setting = SettingMesh(mesh_type="meshzoo_cube_3d", mesh_density_x=4)
    setting.prepare()

    # Act and Assert
    np.testing.assert_allclose(setting.boundary_nodes_volume.sum(), volume)
    np.testing.assert_allclose(
        nph.euclidean_norm_numba(setting.boundary_normals),
        np.ones((len(setting.boundary_normals), 1)),
    )
