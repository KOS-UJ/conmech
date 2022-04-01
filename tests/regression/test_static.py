"""
Created at 21.08.2019
"""
from dataclasses import dataclass

import numpy as np
import pytest

from conmech.problem_solver import Static as StaticProblem
from conmech.problems import Static
from examples.p_slope_contact_law import make_slope_contact_law
from tests.regression.std_boundary import standard_boundary_nodes


@pytest.fixture(params=["direct", "global optimization", "schur"])
def solving_method(request):
    return request.param


def generate_test_suits():
    test_suites = []

    # Simple example

    @dataclass()
    class StaticSetup(Static):
        grid_height: ... = 1
        elements_number: ... = (2, 5)
        mu_coef: ... = 4
        la_coef: ... = 4
        contact_law: ... = make_slope_contact_law(slope=1)

        @staticmethod
        def inner_forces(x, y):
            return np.array([-0.2, -0.2])

        @staticmethod
        def outer_forces(x, y):
            return np.array([0, 0])

        @staticmethod
        def friction_bound(u_nu):
            return 0

        @staticmethod
        def is_contact(x):
            return x[1] == 0

        @staticmethod
        def is_dirichlet(x):
            return x[0] == 0

    setup_m02_m02 = StaticSetup()

    expected_displacement_vector_m02_m02 = [
        [0.0, 0.0],
        [0.04843176, 0.04429129],
        [0.07642598, 0.08750102],
        [0.09064929, 0.13301396],
        [0.09615189, 0.17619656],
        [0.09566479, 0.21772099],
        [0.05373533, 0.22480478],
        [0.00962312, 0.22716041],
        [0.00730758, 0.18136726],
        [0.00104241, 0.13297417],
        [-0.00636696, 0.081923],
        [-0.00867296, 0.03338082],
        [0.0, 0.0],
        [0.0, 0.0]
    ]

    test_suites.append((setup_m02_m02, expected_displacement_vector_m02_m02))

    # p = 0 and opposite forces

    setup_0_02_p_0 = StaticSetup()
    setup_0_02_p_0.contact_law = make_slope_contact_law(slope=0)

    def inner_forces(x, y):
        return np.array([0, 0.2])

    setup_0_02_p_0.inner_forces = inner_forces

    expected_displacement_vector_0_02_p_0 = [
        [0.0, 0.0],
        [-0.11787841, -0.1252073],
        [-0.19322311, -0.31878912],
        [-0.23173236, -0.56035602],
        [-0.24636923, -0.8166048],
        [-0.24898703, -1.07004206],
        [0.0, -1.06985119],
        [0.24898703, -1.07004206],
        [0.24636923, -0.8166048],
        [0.23173236, -0.56035602],
        [0.19322311, -0.31878912],
        [0.11787841, -0.1252073],
        [0.0, 0.0],
        [0.0, 0.0]
    ]

    test_suites.append((setup_0_02_p_0, expected_displacement_vector_0_02_p_0))

    # p = 0

    setup_0_m02_p_0 = StaticSetup()
    setup_0_m02_p_0.contact_law = make_slope_contact_law(slope=0)

    def inner_forces(x, y):
        return np.array([0, -0.2])

    setup_0_m02_p_0.inner_forces = inner_forces

    expected_displacement_vector_0_m02_p_0 = [
        [-v[0], -v[1]] for v in expected_displacement_vector_0_02_p_0
    ]

    test_suites.append((setup_0_m02_p_0, expected_displacement_vector_0_m02_p_0))

    # various changes

    @dataclass()
    class StaticSetup(Static):
        grid_height: ... = 1.37
        elements_number: ... = (2, 5)
        mu_coef: ... = 4.58
        la_coef: ... = 3.33
        contact_law: ... = make_slope_contact_law(slope=2.71)

        @staticmethod
        def inner_forces(x, y):
            return np.array([0, -0.2])

        @staticmethod
        def outer_forces(x, y):
            return np.array([0.3, 0.0])

        @staticmethod
        def friction_bound(u_nu):
            return 0.0

        @staticmethod
        def is_contact(x):
            return x[1] == 0

        @staticmethod
        def is_dirichlet(x):
            return x[0] == 0

    setup_var = StaticSetup()
    expected_displacement_vector_var = [
        [0.0, 0.0],
        [-0.02154956, 0.01364313],
        [-0.04849654, 0.05059958],
        [-0.07590132, 0.0972985],
        [-0.09873572, 0.15498692],
        [-0.12252541, 0.22719522],
        [-0.19937449, 0.26118308],
        [-0.30552747, 0.28092124],
        [-0.27474735, 0.1939756],
        [-0.22880436, 0.13188258],
        [-0.17312159, 0.08296667],
        [-0.10282189, 0.04289061],
        [0.0, 0.0],
        [0.0, 0.0]
    ]

    test_suites.append((setup_var, expected_displacement_vector_var))

    return test_suites


@pytest.mark.parametrize("setup, expected_displacement_vector", generate_test_suits())
def test_direct_solver(solving_method, setup, expected_displacement_vector):
    runner = StaticProblem(setup, solving_method)
    result = runner.solve(initial_displacement=setup.initial_displacement)

    displacement = result.mesh.initial_nodes[:] - result.displaced_points[:]
    std_ids = standard_boundary_nodes(runner.mesh.initial_nodes, runner.mesh.elements)

    # print result
    np.set_printoptions(precision=8, suppress=True)
    print(repr(displacement[std_ids]))

    np.testing.assert_array_almost_equal(
        displacement[std_ids], expected_displacement_vector, decimal=3
    )
