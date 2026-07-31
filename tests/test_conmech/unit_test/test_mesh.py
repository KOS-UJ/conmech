"""
Created at 12.02.2022
"""

import numpy as np
import pytest

from conmech.mesh.boundaries_description import BoundariesDescription
from conmech.mesh.boundaries_factory import BoundariesFactory
from conmech.properties.mesh_description import NestedRectangleMeshDescription
from conmech.mesh.mesh import Mesh
from tests.test_conmech.regression.std_boundary import (
    extract_boundary_paths_from_elements,
)


def test_identify_surfaces():
    elements = np.array(
        [
            [0, 1, 4],
            [0, 1, 7],
            [1, 2, 7],
            [2, 5, 7],
            [2, 5, 8],
            [2, 6, 8],
            [3, 6, 8],
            [3, 4, 6],
            [1, 4, 6],
        ]
    )
    boundary_paths = extract_boundary_paths_from_elements(elements)
    np.testing.assert_array_equal(boundary_paths[0], np.asarray([0, 4, 3, 8, 5, 7, 0]))
    np.testing.assert_array_equal(boundary_paths[1], np.asarray([1, 2, 6, 1]))


def generate_test_suits_old():
    def is_dirichlet(x):
        return x[0] < 4

    expected_dirichlet = [[1, 3, 2]]  # [[2, 3, 1]]

    def is_contact(x):
        return x[1] % 2 == 0

    expected_contact = [[4, 8, 6, 2]]

    expected_neumann = [[1, 7, 9, 4]]

    yield "standard triple", (
        is_dirichlet,
        is_contact,
        expected_dirichlet,
        expected_contact,
        expected_neumann,
    )

    def is_dirichlet(x):
        return x[0] < 5 or x[0] % 2 != 0

    expected_dirichlet = [[2, 3, 1, 7, 9, 4]]

    def is_contact(x):
        return x[0] % 2 == 0

    expected_contact = [[4, 8, 6, 2]]

    expected_neumann = []

    yield "without neumann", (
        is_dirichlet,
        is_contact,
        expected_dirichlet,
        expected_contact,
        expected_neumann,
    )

    def is_dirichlet(x):
        return False

    expected_dirichlet = []

    def is_contact(x):
        return True

    expected_contact = [[1, 3, 2, 6, 8, 4, 9, 7]]  # [[6, 2, 3, 1, 7, 9, 4, 8]]

    expected_neumann = []

    yield "only contact", (
        is_dirichlet,
        is_contact,
        expected_dirichlet,
        expected_contact,
        expected_neumann,
    )

    def is_dirichlet(x):
        return x[0] % 2 != 0

    expected_dirichlet = [[3, 1, 7, 9]]

    def is_contact(x):
        return x[0] % 2 == 0

    expected_contact = [[4, 8, 6, 2]]

    expected_neumann = [[2, 3], [9, 4]]

    yield "double one edge neumann", (
        is_dirichlet,
        is_contact,
        expected_dirichlet,
        expected_contact,
        expected_neumann,
    )

    def is_dirichlet(x):
        return x[0] == 8 or x[0] == 6

    expected_dirichlet = [[8, 6]]

    def is_contact(x):
        return False

    expected_contact = []

    expected_neumann = [[1, 3, 2, 6, 8, 4, 9, 7]]  # [[6, 2, 3, 1, 7, 9, 4, 8]]

    yield "one edge dirichlet beginning-end", (
        is_dirichlet,
        is_contact,
        expected_dirichlet,
        expected_contact,
        expected_neumann,
    )

    def is_dirichlet(x):
        return False

    expected_dirichlet = []

    def is_contact(x):
        return False

    expected_contact = []

    expected_neumann = [[1, 3, 2, 6, 8, 4, 9, 7]]  # [[6, 2, 3, 1, 7, 9, 4, 8]]

    yield "only neumann", (
        is_dirichlet,
        is_contact,
        expected_dirichlet,
        expected_contact,
        expected_neumann,
    )

    def is_dirichlet(x):
        return x[0] < 4

    expected_dirichlet = [[1, 3, 2]]  # [[2, 3, 1]]

    def is_contact(x):
        return False

    expected_contact = []

    expected_neumann = [[1, 7, 9, 4, 8, 6, 2]]

    yield "dirichlet in the middle", (
        is_dirichlet,
        is_contact,
        expected_dirichlet,
        expected_contact,
        expected_neumann,
    )


unordered_nodes = np.asarray([[1.0, 1.0], [0.0, 0.0], [0.0, 2.0], [2.0, 2.0], [2.0, 0.0]])
unordered_elements = np.asarray([[1, 2, 0], [2, 3, 0], [3, 4, 0], [4, 1, 0]])


def generate_test_suits():
    def is_dirichlet(x):
        return x[0] == 0

    def is_contact(x):
        return x[1] == 0

    expected_contact_boundary = np.array([[1, 4]])

    expected_neumann_boundary = np.array([[2, 3], [3, 4]])

    expected_dirichlet_boundary = np.array([[1, 2]])

    yield "standard triple", (
        is_dirichlet,
        is_contact,
        expected_contact_boundary,
        expected_neumann_boundary,
        expected_dirichlet_boundary,
    )


@pytest.mark.parametrize("_test_name_, params", list(generate_test_suits()))
def test_condition_boundaries(_test_name_, params):
    # Arrange
    (
        is_dirichlet,
        is_contact,
        expected_contact_boundary,
        expected_neumann_boundary,
        expected_dirichlet_boundary,
    ) = params
    boundaries_description = BoundariesDescription(contact=is_contact, dirichlet=is_dirichlet)

    # Act
    (
        initial_nodes,
        elements,
        boundaries_data,
    ) = BoundariesFactory.identify_boundaries_and_reorder_nodes(
        unordered_nodes,
        unordered_elements,
        boundaries_description=boundaries_description,
    )

    # Assert
    def unify_edges(boundary):
        return frozenset([frozenset([str(np.sort(node)) for node in edge]) for edge in boundary])

    def compare_surfaces(actual_surfaces, expected_surfaces):
        return unify_edges(initial_nodes[actual_surfaces]) == unify_edges(
            unordered_nodes[expected_surfaces]
        )

    assert compare_surfaces(boundaries_data.contact_boundary, expected_contact_boundary)
    assert compare_surfaces(boundaries_data.neumann_boundary, expected_neumann_boundary)
    assert compare_surfaces(boundaries_data.dirichlet_boundary, expected_dirichlet_boundary)


def test_nested_rectangle_mesh_size_and_counts():
    for cells_per_unit in (1, 2, 4, 8):
        descr = NestedRectangleMeshDescription(
            initial_position=None, cells_per_unit=cells_per_unit, scale=[2, 1]
        )
        mesh = descr.build()
        assert descr.mesh_size == 1.0 / cells_per_unit
        assert len(mesh.nodes) == (2 * cells_per_unit + 1) * (cells_per_unit + 1)
        assert len(mesh.elements) == 4 * cells_per_unit**2

        for axis, extent in enumerate((2.0, 1.0)):
            coordinates = np.unique(mesh.nodes[:, axis])
            np.testing.assert_allclose(
                coordinates, np.linspace(0.0, extent, int(extent * cells_per_unit) + 1), atol=1e-15
            )


def test_nested_rectangle_refinements_are_nested():
    def build(cells_per_unit):
        mesh = NestedRectangleMeshDescription(
            initial_position=None, cells_per_unit=cells_per_unit, scale=[2, 1]
        ).build()
        return np.asarray(mesh.nodes), np.asarray(mesh.elements)

    coarse_nodes, coarse_elements = build(4)
    fine_nodes, fine_elements = build(8)

    # every coarse vertex is a fine vertex
    for node in coarse_nodes:
        assert np.any(np.all(np.isclose(fine_nodes, node, atol=1e-15), axis=1)), node

    # every fine element lies inside exactly one coarse element: its centroid is
    # in that coarse element, and it never straddles a coarse edge
    coarse_centroids = coarse_nodes[coarse_elements].mean(axis=1)
    fine_centroids = fine_nodes[fine_elements].mean(axis=1)
    coarse_area = 0.5 * np.abs(
        np.cross(
            coarse_nodes[coarse_elements[:, 1]] - coarse_nodes[coarse_elements[:, 0]],
            coarse_nodes[coarse_elements[:, 2]] - coarse_nodes[coarse_elements[:, 0]],
        )
    )
    fine_area = 0.5 * np.abs(
        np.cross(
            fine_nodes[fine_elements[:, 1]] - fine_nodes[fine_elements[:, 0]],
            fine_nodes[fine_elements[:, 2]] - fine_nodes[fine_elements[:, 0]],
        )
    )
    np.testing.assert_allclose(coarse_area.sum(), fine_area.sum(), rtol=1e-14)
    assert len(fine_elements) == 4 * len(coarse_elements)

    for centroid in fine_centroids:
        owners = [
            index
            for index, element in enumerate(coarse_elements)
            if _point_in_triangle(centroid, coarse_nodes[element])
        ]
        assert len(owners) == 1, (centroid, owners)

    # each coarse element owns exactly four fine elements
    assert len(coarse_centroids) * 4 == len(fine_centroids)


def _point_in_triangle(point, triangle, tol=1e-12):
    p_0, p_1, p_2 = triangle
    denominator = (p_1[0] - p_0[0]) * (p_2[1] - p_0[1]) - (p_2[0] - p_0[0]) * (p_1[1] - p_0[1])
    lambda_1 = (
        (point[0] - p_0[0]) * (p_2[1] - p_0[1]) - (p_2[0] - p_0[0]) * (point[1] - p_0[1])
    ) / denominator
    lambda_2 = (
        (p_1[0] - p_0[0]) * (point[1] - p_0[1]) - (point[0] - p_0[0]) * (p_1[1] - p_0[1])
    ) / denominator
    return lambda_1 > tol and lambda_2 > tol and lambda_1 + lambda_2 < 1.0 - tol


def test_nested_rectangle_survives_boundary_renumbering():
    description = BoundariesDescription(
        dirichlet=(lambda x: np.isclose(x[1], 0.0), lambda x: np.zeros(x.shape[0])),
        contact=lambda x: np.isclose(x[1], 1.0),
    )

    built = {}
    for cells_per_unit in (4, 8):
        mesh = Mesh(
            mesh_descr=NestedRectangleMeshDescription(
                initial_position=None, cells_per_unit=cells_per_unit, scale=[2, 1]
            ),
            boundaries_description=description,
        )
        built[cells_per_unit] = np.asarray(mesh.nodes)
        assert len(mesh.nodes) == (2 * cells_per_unit + 1) * (cells_per_unit + 1)
        assert len(mesh.elements) == 4 * cells_per_unit**2

    for node in built[4]:
        assert np.any(np.all(np.isclose(built[8], node, atol=1e-15), axis=1)), node
