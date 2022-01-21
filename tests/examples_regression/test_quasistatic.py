"""
Created at 21.08.2019
"""

import numpy as np
import pytest
from dataclasses import dataclass

from conmech.problem_solver import Quasistatic as QuasistaticProblem
from conmech.problems import Quasistatic
from examples.p_slope_contact_law import make_slope_contact_law


@pytest.fixture(params=[  # TODO #28
    # "global optimization",  # TODO #29
    "schur"
])
def solving_method(request):
    return request.param


def generate_test_suits():
    test_suites = []

    # Simple example

    @dataclass()
    class QuasistaticSetup(Quasistatic):
        grid_height: ... = 1
        cells_number: ... = (2, 5)
        inner_forces: ... = np.array([-0.2, -0.2])
        outer_forces: ... = np.array([0, 0])
        mu_coef: ... = 4
        lambda_coef: ... = 4
        th_coef: ... = 4
        ze_coef: ... = 4
        time_step: ... = 0.1
        contact_law: ... = make_slope_contact_law(slope=1e1)

        @staticmethod
        def friction_bound(u_nu):
            return 0

    setup_m02_m02 = QuasistaticSetup()

    expected_displacement_vector_m02_m02 = \
        [-0.03174706, -0.04874277, -0.05728225, -0.06046738, -0.05934763,
         -0.04745009, -0.02026372, -0.03480412, -0.00634097, -0.01233404,
         -0.05049298, -0.04517364, -0.03491524, -0.0188448, -0.04592061,
         -0.05611757, -0.04188783, -0.05420494, -0.03304815, -0.05109881,
         -0.04084441, -0.03832705, -0.03054835, -0.01850842, -0.00648465,
         -0.02791855, -0.04364193, -0.05630732, -0.06519961, -0.07272399,
         -0.05121709, -0.03036478, -0.03437744, -0.00866102, -0.01121569,
         -0.07070694, -0.05903126, -0.04301657, -0.02138356, -0.07706164,
         -0.07295995, -0.06552595, -0.06358101, -0.05057972, -0.0803347,
         -0.08283997, -0.07139739, -0.05772674, -0.03964175, -0.01742878]

    test_suites.append((setup_m02_m02, expected_displacement_vector_m02_m02))

    # p = 0 and opposite forces

    setup_0_02_p_0 = QuasistaticSetup()
    setup_0_02_p_0.contact_law = make_slope_contact_law(slope=0)
    setup_0_02_p_0.inner_forces = np.array([0, 0.2])

    expected_displacement_vector_0_02_p_0 = \
        [0.11383076, 0.18658829, 0.2237752, 0.23790945, 0.24043737,
         0.10311252, -0.07648543, 0.07648539, -0.02603917, 0.02603916,
         -0.00000004, -0.00000003, -0.00000002, -0.00000001, -0.11970527,
         0.11970519, -0.1157959, 0.11579583, -0.10311257, -0.00000004,
         -0.24043746, -0.23790954, -0.22377527, -0.18658834, -0.11383078,
         0.120908, 0.30784272, 0.54111482, 0.78856463, 1.03329948,
         0.41722515, 0.19887081, 0.19887081, 0.04189106, 0.04189106,
         0.7874323, 0.53698504, 0.29953967, 0.10155982, 0.91078986,
         0.91078985, 0.66226037, 0.66226037, 0.41722515, 1.03311517,
         1.03329948, 0.78856464, 0.54111483, 0.30784273, 0.12090801]

    test_suites.append((setup_0_02_p_0, expected_displacement_vector_0_02_p_0))

    # p = 0

    setup_0_m02_p_0 = QuasistaticSetup()
    setup_0_m02_p_0.contact_law = make_slope_contact_law(slope=0)
    setup_0_m02_p_0.inner_forces = np.array([0, -0.2])
    expected_displacement_vector_0_m02_p_0 = [-v for v in expected_displacement_vector_0_02_p_0]
    test_suites.append((setup_0_m02_p_0, expected_displacement_vector_0_m02_p_0))

    # various changes

    @dataclass()
    class QuasistaticSetup(Quasistatic):
        grid_height: ... = 1.37
        cells_number: ... = (2, 5)
        inner_forces: ... = np.array([0, -0.2])
        outer_forces: ... = np.array([0.3, 0.0])
        mu_coef: ... = 4.58
        lambda_coef: ... = 3.33
        th_coef: ... = 2.11
        ze_coef: ... = 4.99
        time_step: ... = 0.1
        contact_law: ... = make_slope_contact_law(slope=2.71)

        @staticmethod
        def friction_bound(u_nu):
            return 0.0

    setup_var = QuasistaticSetup()
    expected_displacement_vector_var = \
        [-0.02607508, -0.03693133, -0.03541439, -0.02747097, -0.01291643,
         0.03057441, 0.11811058, 0.01390255, 0.04038013, 0.00415804,
         0.14088622, 0.11491156, 0.08150496, 0.04103966, 0.24646681,
         0.06452929, 0.21760316, 0.04774937, 0.17504007, 0.16050503,
         0.35617261, 0.32973286, 0.28372653, 0.21878206, 0.12898925,
         -0.05328279, -0.15587498, -0.29434243, -0.45490567, -0.6276158,
         -0.22646547, -0.11105349, -0.10056004, -0.02139806, -0.01885423,
         -0.46991231, -0.30505151, -0.16286354, -0.05168829, -0.56553781,
         -0.55060983, -0.39162963, -0.37967762, -0.23799145, -0.64941679,
         -0.66290604, -0.48007335, -0.31781755, -0.17832825, -0.07212695]

    test_suites.append((setup_var, expected_displacement_vector_var))

    return test_suites


@pytest.mark.parametrize('setup, expected_displacement_vector', generate_test_suits())
def test_global_optimization_solver(solving_method, setup, expected_displacement_vector):
    runner = QuasistaticProblem(setup, solving_method)
    results = runner.solve(n_steps=32)
    displacement_vector = results[-1].displacement.T.reshape(1, -1)[0]
    np.testing.assert_array_almost_equal(displacement_vector, expected_displacement_vector, decimal=5)
