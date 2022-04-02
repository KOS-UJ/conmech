"""
Created at 21.08.2019
"""

from dataclasses import dataclass

import numpy as np
import pytest

from conmech.problem_solver import Dynamic as DynamicProblem
from conmech.problems import Dynamic
from examples.p_slope_contact_law import make_slope_contact_law
from tests.regression.std_boundary import standard_boundary_nodes


@pytest.fixture(
    params=[  # TODO #28
        "global optimization",
        "schur"
    ]
)
def solving_method(request):
    return request.param


def generate_test_suits():
    test_suites = []

    # Simple example

    @dataclass()
    class DynamicSetup(Dynamic):
        grid_height: ... = 1
        elements_number: ... = (2, 5)
        mu_coef: ... = 4
        la_coef: ... = 4
        th_coef: ... = 4
        ze_coef: ... = 4
        time_step: ... = 0.1
        contact_law: ... = make_slope_contact_law(slope=1e1)

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

    setup_m02_m02 = DynamicSetup()

    expected_displacement_vector_m02_m02 = [
        [0.0, 0.0],
        [0.03194299, 0.02783408],
        [0.04913394, 0.04305829],
        [0.05786617, 0.05507215],
        [0.06119214, 0.06324329],
        [0.06012923, 0.07002092],
        [0.05260607, 0.077619],
        [0.04304498, 0.08011945],
        [0.04043799, 0.06935685],
        [0.03238124, 0.05635081],
        [0.01987252, 0.0388462],
        [0.00722258, 0.01704675],
        [0.0, 0.0],
        [0.0, 0.0]
    ]

    test_suites.append((setup_m02_m02, expected_displacement_vector_m02_m02))

    # p = 0 and opposite forces

    setup_0_02_p_0 = DynamicSetup()
    setup_0_02_p_0.contact_law = make_slope_contact_law(slope=0)

    def inner_forces(x, y):
        return np.array([0, 0.2])

    setup_0_02_p_0.inner_forces = inner_forces

    expected_displacement_vector_0_02_p_0 = [
        [0.0, 0.0],
        [-0.10204928, -0.11029064],
        [-0.16607854, -0.27796998],
        [-0.19804137, -0.48509912],
        [-0.20984607, -0.70344499],
        [-0.21187304, -0.91900424],
        [0.00000004, -0.91885112],
        [0.21187312, -0.91900425],
        [0.20984614, -0.70344499],
        [0.19804144, -0.48509913],
        [0.16607859, -0.27796999],
        [0.10204931, -0.11029065],
        [0.0, 0.0],
        [0.0, 0.0]
    ]

    test_suites.append((setup_0_02_p_0, expected_displacement_vector_0_02_p_0))

    # p = 0

    setup_0_m02_p_0 = DynamicSetup()
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
    class DynamicSetup(Dynamic):
        grid_height: ... = 1.37
        elements_number: ... = (2, 5)
        mu_coef: ... = 4.58
        la_coef: ... = 3.33
        th_coef: ... = 2.11
        ze_coef: ... = 4.99
        time_step: ... = 0.1
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

    setup_var = DynamicSetup()
    expected_displacement_vector_var = [
        [0.0, 0.0],
        [0.00805403, 0.04399692],
        [0.00357147, 0.13577263],
        [-0.01047404, 0.26004496],
        [-0.0274148, 0.40577722],
        [-0.04940713, 0.56621234],
        [-0.21158738, 0.59684774],
        [-0.40204546, 0.6157323],
        [-0.36922423, 0.44259321],
        [-0.31434358, 0.29464635],
        [-0.24006106, 0.1695022],
        [-0.14068317, 0.07419011],
        [0.0, 0.0],
        [0.0, 0.0]
    ]

    test_suites.append((setup_var, expected_displacement_vector_var))

    return test_suites


@pytest.mark.parametrize("setup, expected_displacement_vector", generate_test_suits())
def test_global_optimization_solver(
        solving_method, setup, expected_displacement_vector
):
    runner = DynamicProblem(setup, solving_method)
    results = runner.solve(n_steps=32,
                           initial_displacement=setup.initial_displacement,
                           initial_velocity=setup.initial_velocity)

    displacement = results[-1].mesh.initial_nodes[:] - results[-1].displaced_points[:]
    std_ids = standard_boundary_nodes(runner.mesh.initial_nodes, runner.mesh.elements)

    # print result
    np.set_printoptions(precision=8, suppress=True)
    print(repr(displacement[std_ids]))

    np.testing.assert_array_almost_equal(
        displacement[std_ids], expected_displacement_vector, decimal=3
    )
