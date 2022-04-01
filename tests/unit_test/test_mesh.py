"""
Created at 12.02.2022
"""

import numpy as np
import pytest

from conmech.features.boundaries import identify_surfaces, get_boundaries


def test_identify_surfaces():
    elements = [[0, 1, 4],
                [0, 1, 7],
                [1, 2, 7],
                [2, 5, 7],
                [2, 5, 8],
                [2, 6, 8],
                [3, 6, 8],
                [3, 4, 6],
                [1, 4, 6]]
    vertex_num = 9
    surfaces = identify_surfaces(elements, vertex_num)
    np.testing.assert_array_equal(surfaces[0], np.asarray([0, 4, 3, 8, 5, 7, 0]))
    np.testing.assert_array_equal(surfaces[1], np.asarray([1, 2, 6, 1]))


def generate_test_suits():
    def is_dirichlet(x):
        return x[0] < 4

    expected_dirichlet = [[2, 3, 1]]

    def is_contact(x):
        return x[1] % 2 == 0

    expected_contact = [[4, 8, 6, 2]]

    expected_neumann = [[1, 7, 9, 4]]

    yield "standard triple", \
          (is_dirichlet, is_contact, expected_dirichlet, expected_contact, expected_neumann)

    def is_dirichlet(x):
        return x[0] < 5 or x[0] % 2 != 0

    expected_dirichlet = [[2, 3, 1, 7, 9, 4]]

    def is_contact(x):
        return x[0] % 2 == 0

    expected_contact = [[4, 8, 6, 2]]

    expected_neumann = []

    yield "without neumann", \
          (is_dirichlet, is_contact, expected_dirichlet, expected_contact, expected_neumann)

    def is_dirichlet(x):
        return False

    expected_dirichlet = []

    def is_contact(x):
        return True

    expected_contact = [[6, 2, 3, 1, 7, 9, 4, 8]]

    expected_neumann = []

    yield "only contact", \
          (is_dirichlet, is_contact, expected_dirichlet, expected_contact, expected_neumann)

    def is_dirichlet(x):
        return x[0] % 2 != 0

    expected_dirichlet = [[3, 1, 7, 9]]

    def is_contact(x):
        return x[0] % 2 == 0

    expected_contact = [[4, 8, 6, 2]]

    expected_neumann = [[2, 3], [9, 4]]

    yield "double one edge neumann", \
          (is_dirichlet, is_contact, expected_dirichlet, expected_contact, expected_neumann)

    def is_dirichlet(x):
        return x[0] == 8 or x[0] == 6

    expected_dirichlet = [[8, 6]]

    def is_contact(x):
        return False

    expected_contact = []

    expected_neumann = [[6, 2, 3, 1, 7, 9, 4, 8]]

    yield "one edge dirichlet beginning-end", \
          (is_dirichlet, is_contact, expected_dirichlet, expected_contact, expected_neumann)

    def is_dirichlet(x):
        return False

    expected_dirichlet = []

    def is_contact(x):
        return False

    expected_contact = []

    expected_neumann = [[6, 2, 3, 1, 7, 9, 4, 8]]

    yield "only neumann", \
          (is_dirichlet, is_contact, expected_dirichlet, expected_contact, expected_neumann)

    def is_dirichlet(x):
        return x[0] < 4

    expected_dirichlet = [[2, 3, 1]]

    def is_contact(x):
        return False

    expected_contact = []

    expected_neumann = [[1, 7, 9, 4, 8, 6, 2]]

    yield "dirichlet in the middle", \
          (is_dirichlet, is_contact, expected_dirichlet, expected_contact, expected_neumann)


@pytest.mark.parametrize('_test_name_, params', list(generate_test_suits()))
def test_condition_boundaries(_test_name_, params):
    # Arrange
    is_dirichlet, is_contact, expected_dirichlet, expected_contact, expected_neumann = params
    vertices = np.asarray([
        [0, 0], [1, 1], [2, 2], [3, 3], [4, 4], [5, 5], [6, 6], [7, 7], [8, 8], [9, 9]
    ])
    boundaries = [np.asarray([6, 2, 3, 1, 7, 9, 4, 8])]

    # Act
    contact, dirichlet, neumann = get_boundaries(
        is_contact, is_dirichlet, boundaries, vertices)

    # Assert
    np.testing.assert_array_equal(dirichlet, expected_dirichlet)
    np.testing.assert_array_equal(contact, expected_contact)
    np.testing.assert_array_equal(neumann, expected_neumann)
