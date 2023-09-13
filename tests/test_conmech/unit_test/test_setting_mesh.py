import numpy as np
import pytest

from conmech.helpers import nph
from conmech.mesh import mesh
from conmech.mesh.boundaries_description import BoundariesDescription
from conmech.mesh.mesh import Mesh
from conmech.properties.mesh_description import RectangleMeshDescription, CubeMeshDescription
from conmech.simulations.problem_solver import Body


@pytest.mark.parametrize("scale_x, scale_y", ((1, 1), (2, 3), (5, 1)))
def test_boundary_nodes_data_2d(scale_x, scale_y):
    # Arrange
    boundaries_description: ... = BoundariesDescription(
        contact=lambda x: True, dirichlet=lambda x: False
    )
    mesh_descr = RectangleMeshDescription(
        initial_position=None,
        max_element_perimeter=(np.min([scale_x, scale_y]) / 3),
        scale=[scale_x, scale_y],
    )
    mesh = Mesh(
        mesh_descr=mesh_descr,
        boundaries_description=boundaries_description,
    )

    # Act and Assert
    boundary_normals = mesh.boundaries.surface_normals
    np.testing.assert_allclose(
        nph.euclidean_norm_numba(boundary_normals),
        np.ones(len(boundary_normals)),
    )


def test_boundary_nodes_data_3d():
    # Arrange
    boundaries_description: ... = BoundariesDescription(
        contact=lambda x: True, dirichlet=lambda x: False
    )
    mesh_descr = CubeMeshDescription(initial_position=None)
    mesh = Mesh(
        mesh_descr=mesh_descr,
        boundaries_description=boundaries_description,
    )

    # Act and Assert
    boundary_normals = mesh.boundaries.surface_normals
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
