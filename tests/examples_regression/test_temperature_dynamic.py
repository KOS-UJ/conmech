"""
Created at 21.08.2019
"""

import numpy as np
import pytest
from dataclasses import dataclass

from conmech.problem_solver import TDynamic as TDynamicProblem
from conmech.problems import Dynamic
from examples.p_slope_contact_law import make_slope_contact_law
from tests.examples_regression.std_boundary import standard_boundary_nodes


@pytest.fixture(params=[  # TODO #28
    # "global optimization",  TODO #29
    "schur"
])
def solving_method(request):
    return request.param


def make_slope_contact_law_temp(slope):
    class TPSlopeContactLaw(make_slope_contact_law(slope=slope)):
        @staticmethod
        def h_nu(uN, t):
            g_t = 10.7 + t * 0.02
            if uN > g_t:
                return 100. * (uN - g_t)
            return 0

        @staticmethod
        def h_tau(uN, t):
            g_t = 10.7 + t * 0.02
            if uN > g_t:
                return 10. * (uN - g_t)
            return 0

        @staticmethod
        def h_temp(vTnorm):
            return 0.1 * vTnorm

    return TPSlopeContactLaw


def generate_test_suits():
    test_suites = []

    # Simple example

    @dataclass()
    class DynamicSetup(Dynamic):
        grid_height: ... = 1
        cells_number: ... = (2, 5)
        mu_coef: ... = 4
        la_coef: ... = 4
        th_coef: ... = 4
        ze_coef: ... = 4
        time_step: ... = 0.1
        contact_law: ... = make_slope_contact_law_temp(1e1)

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
        def is_contact(x, y):
            return y == 0

        @staticmethod
        def is_dirichlet(x, y):
            return x == 0

    setup_m02_m02 = DynamicSetup()

    expected_displacement_vector_m02_m02 = np.asarray([
        [0., 0.],
        [0.01421449, 0.01527654],
        [0.02183647, 0.03203525],
        [0.02500916, 0.04856747],
        [0.0259099, 0.06356947],
        [0.02560049, 0.07813952],
        [0.01063955, 0.07989915],
        [-0.00500671, 0.08043558],
        [-0.00548251, 0.06452265],
        [-0.00685184, 0.047779],
        [-0.00796572, 0.02975652],
        [-0.00635081, 0.01246021],
        [0., 0.],
        [0., 0.]])
    expected_temperature_vector_m02_m02 = np.asarray(
        [0., 0.00653853, 0.01061051, 0.01275503, 0.01360124,
         0.0137999, 0.01160837, 0.01083679, 0.01065018, 0.00980904,
         0.00769099, 0.00399129, 0., 0.])

    test_suites.append(
        (setup_m02_m02, expected_displacement_vector_m02_m02, expected_temperature_vector_m02_m02))

    # p = 0 and opposite forces

    setup_0_02_p_0 = DynamicSetup()
    setup_0_02_p_0.contact_law = make_slope_contact_law_temp(0)

    def inner_forces(x, y):
        return np.array([0, 0.2])

    setup_0_02_p_0.inner_forces = inner_forces

    expected_displacement_vector_0_02_p_0 = np.asarray([
        [0., 0.],
        [-0.02352764, -0.02462197],
        [-0.03875654, -0.0632376],
        [-0.04664045, -0.11167183],
        [-0.0497121, -0.16315532],
        [-0.05034563, -0.21410253],
        [-0.00029333, -0.21415888],
        [0.04976176, -0.21428449],
        [0.04930993, -0.16332813],
        [0.04640432, -0.11182045],
        [0.03865392, -0.06334662],
        [0.02351049, -0.02467719],
        [0., 0.],
        [0., 0.]])
    expected_temperature_vector_0_02_p_0 = np.asarray(
        [0., 0.01345895, 0.02718391, 0.03769502, 0.04415301,
         0.04631775, 0.04032226, 0.03878766, 0.03713804, 0.03224875,
         0.02437599, 0.01398369, 0., 0.])

    test_suites.append(
        (setup_0_02_p_0, expected_displacement_vector_0_02_p_0,
         expected_temperature_vector_0_02_p_0))

    # p = 0

    setup_0_m02_p_0 = DynamicSetup()
    setup_0_m02_p_0.contact_law = make_slope_contact_law_temp(0)

    def inner_forces(x, y):
        return np.array([0, -0.2])

    setup_0_m02_p_0.inner_forces = inner_forces
    expected_displacement_vector_0_m02_p_0 = [-v for v in expected_displacement_vector_0_02_p_0]
    expected_temperature_vector_0_m02_p_0 = np.asarray(
        [0., 0.01692873, 0.02963103, 0.03895211, 0.04455073,
         0.04639868, 0.04017245, 0.03840349, 0.03644976, 0.03073947,
         0.02174105, 0.01041326, 0., 0.])
    test_suites.append(
        (setup_0_m02_p_0, expected_displacement_vector_0_m02_p_0,
         expected_temperature_vector_0_m02_p_0))

    # various changes

    @dataclass()
    class DynamicSetup(Dynamic):
        grid_height: ... = 1.37
        cells_number: ... = (2, 5)
        mu_coef: ... = 4.58
        la_coef: ... = 3.33
        th_coef: ... = 2.11
        ze_coef: ... = 4.99
        time_step: ... = 0.1
        contact_law: ... = make_slope_contact_law_temp(2.71)

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
        def is_contact(x, y):
            return y == 0

        @staticmethod
        def is_dirichlet(x, y):
            return x == 0

    setup_var = DynamicSetup()
    expected_displacement_vector_var = np.asarray([
        [0., 0.],
        [0.03280209, 0.03595636],
        [0.05390165, 0.10433753],
        [0.0642184, 0.19502387],
        [0.067088, 0.29652378],
        [0.06548885, 0.40042331],
        [-0.03693406, 0.40267814],
        [-0.14223637, 0.40440701],
        [-0.13602867, 0.29999152],
        [-0.12112019, 0.19929007],
        [-0.09523431, 0.10930828],
        [-0.05564706, 0.04098661],
        [0., 0.],
        [0., 0.]])

    expected_temperature_vector_var = np.asarray(
        [0., 0.01290469, 0.02102734, 0.02655083, 0.02976221,
         0.03080907, 0.02203393, 0.01868357, 0.0174178, 0.01384119,
         0.00860922, 0.00284346, 0., 0.])

    test_suites.append(
        (setup_var, expected_displacement_vector_var, expected_temperature_vector_var))

    return test_suites


@pytest.mark.parametrize('setup, expected_displacement_vector, expected_temperature_vector',
                         generate_test_suits())
def test_global_optimization_solver(
        solving_method, setup, expected_displacement_vector, expected_temperature_vector):
    runner = TDynamicProblem(setup, solving_method)
    results = runner.solve(n_steps=32)

    std_ids = standard_boundary_nodes(runner.mesh.initial_nodes, runner.mesh.cells)
    displacement = results[-1].mesh.initial_nodes[:] - results[-1].displaced_points[:]
    temperature = np.zeros(len(results[-1].mesh.initial_nodes))
    temperature[:len(results[-1].temperature)] = results[-1].temperature

    # print result
    np.set_printoptions(precision=8, suppress=True)
    print(repr(displacement[std_ids]))
    print(repr(temperature[std_ids]))

    np.testing.assert_array_almost_equal(
        displacement[std_ids], expected_displacement_vector, decimal=3
    )
    np.testing.assert_array_almost_equal(
        temperature[std_ids], expected_temperature_vector, decimal=3
    )
