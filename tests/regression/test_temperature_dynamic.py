"""
Created at 21.08.2019
"""

from dataclasses import dataclass

import numpy as np
import pytest

from conmech.problem_solver import TDynamic as TDynamicProblem
from conmech.problems import Dynamic
from examples.p_slope_contact_law import make_slope_contact_law
from tests.regression.std_boundary import standard_boundary_nodes


@pytest.fixture(params=[  # TODO #28
    "global optimization",
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
        elements_number: ... = (2, 5)
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
        def is_contact(x):
            return x[1] == 0

        @staticmethod
        def is_dirichlet(x):
            return x[0] == 0

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
        [0., 0.00453681, 0.00656749, 0.00721468, 0.00718777,
         0.00710525, 0.0070813, 0.00705799, 0.00709502, 0.00686151,
         0.00556973, 0.0027622, 0., 0.])

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
        [-0.02095715, -0.02228123],
        [-0.03429104, -0.05672799],
        [-0.04103037, -0.099573],
        [-0.04354969, -0.14488408],
        [-0.04398988, -0.18965005],
        [0.00000001, -0.18961762],
        [0.0439899, -0.18965005],
        [0.04354971, -0.14488408],
        [0.04103039, -0.099573],
        [0.03429105, -0.05672799],
        [0.02095716, -0.02228123],
        [0., 0.],
        [0., 0.]])
    expected_temperature_vector_0_02_p_0 = np.asarray(
        [0., -0.0024217, -0.0017286, -0.00093245, -0.00035858,
         -0.00014683, -0.00000011, 0.00014659, 0.00035835, 0.00093224,
         0.00172843, 0.00242159, 0., 0.])

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
    expected_temperature_vector_0_m02_p_0 = [-v for v in expected_temperature_vector_0_02_p_0]
    test_suites.append(
        (setup_0_m02_p_0, expected_displacement_vector_0_m02_p_0,
         expected_temperature_vector_0_m02_p_0))

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
        def is_contact(x):
            return x[1] == 0

        @staticmethod
        def is_dirichlet(x):
            return x[0] == 0

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
        [0., -0.0014602, -0.00670457, -0.01099723, -0.01357647,
         -0.01442459, -0.01458096, -0.01473978, -0.01469881, -0.01429765,
         -0.01281657, -0.00928493, 0., 0.])

    test_suites.append(
        (setup_var, expected_displacement_vector_var, expected_temperature_vector_var))

    return test_suites


@pytest.mark.parametrize('setup, expected_displacement_vector, expected_temperature_vector',
                         generate_test_suits())
def test_global_optimization_solver(
        solving_method, setup, expected_displacement_vector, expected_temperature_vector):
    runner = TDynamicProblem(setup, solving_method)
    results = runner.solve(n_steps=32,
                           initial_displacement=setup.initial_displacement,
                           initial_velocity=setup.initial_velocity,
                           initial_temperature=setup.initial_temperature)

    std_ids = standard_boundary_nodes(runner.mesh.initial_nodes, runner.mesh.elements)
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
