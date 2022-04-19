import numpy as np
import pytest

from conmech.helpers import nph
from conmech.mesh import mesh
from conmech.properties.mesh_properties import MeshProperties
from conmech.properties.schedule import Schedule
from conmech.state.body_position import BodyPosition


@pytest.mark.parametrize("scale_x, scale_y", ((1, 1), (2, 3), (5, 1)))
def test_boundary_nodes_data_2d(scale_x, scale_y):
    # Arrange
    volume = 2 * (scale_x + scale_y)
    setting = BodyPosition(
        mesh_prop=MeshProperties(
            dimension=2,
            mesh_type="meshzoo_rectangle",
            mesh_density=[3, 3],
            scale=[scale_x, scale_y],
        ),
        schedule=Schedule(1),
        normalize_by_rotation=True,
    )

    # Act and Assert
    np.testing.assert_allclose(setting.get_surface_per_boundary_node().sum(), volume)
    boundary_normals = setting.get_boundary_normals()
    np.testing.assert_allclose(
        nph.euclidean_norm_numba(boundary_normals),
        np.ones(len(boundary_normals)),
    )


def test_boundary_nodes_data_3d():
    # Arrange
    volume = 6
    setting = BodyPosition(
        mesh_prop=MeshProperties(
            dimension=2, mesh_type="meshzoo_cube_3d", mesh_density=[4], scale=[1]
        ),
        schedule=Schedule(1),
        normalize_by_rotation=True,
    )

    # Act and Assert
    np.testing.assert_allclose(setting.get_surface_per_boundary_node().sum(), volume)
    boundary_normals = setting.get_boundary_normals()
    np.testing.assert_allclose(
        nph.euclidean_norm_numba(boundary_normals),
        np.ones(len(boundary_normals)),
    )


def test_remove_unconnected_nodes():
    # Arrange
    nodes = np.array(
        [
            [0.1, 1.2],
            [1.1, 1.2],
            [2.1, 1.2],
            [3.1, 1.2],
            [4.1, 1.2],
            [5.1, 1.2],
            [6.1, 1.2],
        ]
    )
    elements = np.array([[4, 2], [2, 5], [4, 5]])

    # Act
    cleaned_nodes, cleaned_elements = mesh.remove_unconnected_nodes_numba(nodes, elements)

    # Assert
    expected_nodes = np.array([[2.1, 1.2], [4.1, 1.2], [5.1, 1.2]])
    expected_elements = np.array([[1, 0], [0, 2], [1, 2]])
    np.testing.assert_array_equal(expected_nodes, cleaned_nodes)
    np.testing.assert_array_equal(expected_elements, cleaned_elements)
