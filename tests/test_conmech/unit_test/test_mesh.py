"""
Created at 12.02.2022
"""

import numpy as np
import pytest
from conmech.mesh.boundaries_factory import BoundariesFactory
from tests.test_conmech.regression.std_boundary import extract_boundary_paths_from_elements


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

    # Act
    (
        initial_nodes,
        elements,
        boundaries_data,
    ) = BoundariesFactory.identify_boundaries_and_reorder_nodes(
        unordered_nodes, unordered_elements, is_dirichlet, is_contact
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
