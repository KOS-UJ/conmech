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
    # "global optimization",  TODO #29
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
        [-0.03158117, -0.04874298, -0.05737632, -0.06060299, -0.05949624,
         -0.04741211, -0.02016569, -0.03461692, -0.00635714, -0.01234932,
         -0.05039065, -0.04505924, -0.03475237, -0.01877504, -0.04569489,
         -0.05613981, -0.0416697, -0.05421077, -0.03285257, -0.05099982,
         -0.0404911, -0.03797851, -0.03023616, -0.01834908, -0.00653857,
         -0.02658475, -0.04300522, -0.05588787, -0.06503391, -0.07280835,
         -0.05064518, -0.0296611, -0.03348133, -0.00834663, -0.0107092,
         -0.07051681, -0.05856661, -0.04227247, -0.02064332, -0.07699962,
         -0.0729089, -0.06518792, -0.06326734, -0.04995989, -0.08040678,
         -0.08290835, -0.07120304, -0.05724797, -0.03895059, -0.01684964]

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
        [-0.01988354, -0.02422958, -0.01606535, -0.0025166, 0.0179486,
         0.04869326, 0.13404873, 0.02442506, 0.04463835, 0.00681826,
         0.17128634, 0.13895638, 0.09794086, 0.04816707, 0.2850649,
         0.09482256, 0.2500336, 0.07233628, 0.19984991, 0.19652605,
         0.40327365, 0.3710444, 0.31733301, 0.24364559, 0.14324279,
         -0.05033902, -0.15519432, -0.29505867, -0.45770757, -0.63482323,
         -0.22889975, -0.11684529, -0.1007449, -0.02284018, -0.01841484,
         -0.47831459, -0.31073366, -0.16706123, -0.05323697, -0.57937321,
         -0.55901785, -0.40082219, -0.38408345, -0.24535934, -0.66392375,
         -0.68231349, -0.49267371, -0.32825663, -0.18774349, -0.08060115]

    test_suites.append((setup_var, expected_displacement_vector_var))

    return test_suites


@pytest.mark.parametrize('setup, expected_displacement_vector', generate_test_suits())
def test_global_optimization_solver(solving_method, setup, expected_displacement_vector):
    runner = QuasistaticProblem(setup, solving_method)
    results = runner.solve(n_steps=32)
    displacement_vector = results[-1].displacement.T.reshape(1, -1)[0]
    np.testing.assert_array_almost_equal(displacement_vector, expected_displacement_vector, decimal=5)
