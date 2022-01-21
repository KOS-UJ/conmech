"""
Created at 21.08.2019
"""
from dataclasses import dataclass

import numpy as np
import pytest

from conmech.problem_solver import Static as StaticProblem
from conmech.problems import Static
from examples.p_slope_contact_law import make_slope_contact_law


@pytest.fixture(params=[
    "direct",
    "global optimization",
    "schur"
])
def solving_method(request):
    return request.param


def generate_test_suits():
    test_suites = []

    # Simple example

    @dataclass()
    class StaticSetup(Static):
        grid_height: ... = 1
        cells_number: ... = (2, 5)
        inner_forces: ... = np.array([-0.2, -0.2])
        outer_forces: ... = np.array([0, 0])
        mu_coef: ... = 4
        lambda_coef: ... = 4
        contact_law: ... = make_slope_contact_law(slope=1)

        @staticmethod
        def friction_bound(u_nu):
            return 0

    setup_m02_m02 = StaticSetup()

    expected_displacement_vector_m02_m02 = \
        [-0.04843176, -0.07642598, -0.09064929, -0.09615189, -0.09566479,
         -0.06363834, -0.01052346, -0.04646055, -0.00302651, -0.01631055,
         -0.05291791, -0.04715695, -0.03620365, -0.01943901, -0.03164418,
         -0.07544455, -0.0278443, -0.0726739, -0.02021358, -0.05373533,
         -0.00962312, -0.00730758, -0.00104241, 0.00636696, 0.00867296,
         -0.04429129, -0.08750102, -0.13301396, -0.17619656, -0.21772099,
         -0.10993322, -0.05740487, -0.06193063, -0.01406892, -0.01668643,
         -0.18083049, -0.13417542, -0.08459416, -0.03486749, -0.20424883,
         -0.2005721, -0.1579207, -0.15660724, -0.10849453, -0.22480478,
         -0.22716041, -0.18136726, -0.13297417, -0.081923, -0.03338082]

    test_suites.append((setup_m02_m02, expected_displacement_vector_m02_m02))

    # p = 0 and opposite forces

    setup_0_02_p_0 = StaticSetup()
    setup_0_02_p_0.contact_law = make_slope_contact_law(slope=0)
    setup_0_02_p_0.inner_forces = np.array([0, 0.2])

    expected_displacement_vector_0_02_p_0 = \
        [0.11787841, 0.19322311, 0.23173236, 0.24636923, 0.24898703,
         0.10677908, -0.07920512, 0.07920512, -0.02696508, 0.02696508,
         -0., -0., -0., -0., -0.12396178,
         0.12396178, -0.1199134, 0.1199134, -0.10677908, -0.,
         -0.24898703, -0.24636923, -0.23173236, -0.19322311, -0.11787841,
         0.1252073, 0.31878912, 0.56035602, 0.8166048, 1.07004206,
         0.43206101, 0.20594233, 0.20594233, 0.04338063, 0.04338063,
         0.81543219, 0.55607938, 0.31019082, 0.10517111, 0.94317617,
         0.94317617, 0.68580932, 0.68580932, 0.43206101, 1.06985119,
         1.07004206, 0.8166048, 0.56035602, 0.31878912, 0.1252073]

    test_suites.append((setup_0_02_p_0, expected_displacement_vector_0_02_p_0))

    # p = 0

    setup_0_m02_p_0 = StaticSetup()
    setup_0_m02_p_0.contact_law = make_slope_contact_law(slope=0)
    setup_0_m02_p_0.inner_forces = np.array([0, -0.2])
    expected_displacement_vector_0_m02_p_0 = [-v for v in expected_displacement_vector_0_02_p_0]
    test_suites.append((setup_0_m02_p_0, expected_displacement_vector_0_m02_p_0))

    # various changes

    @dataclass()
    class StaticSetup(Static):
        grid_height: ... = 1.37
        cells_number: ... = (2, 5)
        inner_forces: ... = np.array([0, -0.2])
        outer_forces: ... = np.array([0.3, 0.0])
        mu_coef: ... = 4.58
        lambda_coef: ... = 3.33
        contact_law: ... = make_slope_contact_law(slope=2.71)

        @staticmethod
        def friction_bound(u_nu):
            return 0.0

    setup_var = StaticSetup()
    expected_displacement_vector_var = \
        [0.02154956, 0.04849654, 0.07590132, 0.09873572, 0.12252541,
         0.09046129, 0.10615159, 0.05368447, 0.03566161, 0.01637239,
         0.17316701, 0.14016189, 0.0987414, 0.04865871, 0.23759568,
         0.14735179, 0.20432764, 0.12137374, 0.16031448, 0.19937449,
         0.30552747, 0.27474735, 0.22880436, 0.17312159, 0.10282189,
         -0.01364313, -0.05059958, -0.0972985, -0.15498692, -0.22719522,
         -0.08204104, -0.05265191, -0.03691015, -0.01132087, -0.00714311,
         -0.17948923, -0.11646834, -0.06662733, -0.02382621, -0.22927884,
         -0.2063498, -0.1548714, -0.13679878, -0.09875027, -0.26118308,
         -0.28092124, -0.1939756, -0.13188258, -0.08296667, -0.04289061]

    test_suites.append((setup_var, expected_displacement_vector_var))

    return test_suites


@pytest.mark.parametrize('setup, expected_displacement_vector', generate_test_suits())
def test_direct_solver(solving_method, setup, expected_displacement_vector):
    runner = StaticProblem(setup, solving_method)
    result = runner.solve()
    displacement_vector = result.displacement.T.reshape(1, -1)[0]
    np.testing.assert_array_almost_equal(
        displacement_vector, expected_displacement_vector, decimal=3)
