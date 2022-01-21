"""
Created at 21.08.2019
"""

import numpy as np
import pytest
from dataclasses import dataclass

from conmech.problem_solver import Dynamic as DynamicProblem
from conmech.problems import Dynamic
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
    class DynamicSetup(Dynamic):
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

    setup_m02_m02 = DynamicSetup()

    expected_displacement_vector_m02_m02 = \
        [-0.03194297, -0.04913391, -0.05786613, -0.0611921, -0.06012919,
         -0.0482382, -0.02113984, -0.03529847, -0.00661447, -0.01248624,
         -0.05192711, -0.04639885, -0.03580917, -0.01929203, -0.0477569,
         -0.05724707, -0.04356064, -0.05521215, -0.03439656, -0.05260604,
         -0.04304496, -0.04043798, -0.03238123, -0.01987252, -0.00722257,
         -0.02783407, -0.04305827, -0.05507212, -0.06324326, -0.07002088,
         -0.05030119, -0.0298892, -0.03403933, -0.00855841, -0.01118334,
         -0.06871681, -0.05775388, -0.04237052, -0.02120613, -0.07469563,
         -0.07061826, -0.06386439, -0.06197502, -0.0495754, -0.07761896,
         -0.0801194, -0.06935682, -0.05635078, -0.03884618, -0.01704675]

    test_suites.append((setup_m02_m02, expected_displacement_vector_m02_m02))

    # p = 0 and opposite forces

    setup_0_02_p_0 = DynamicSetup()
    setup_0_02_p_0.contact_law = make_slope_contact_law(slope=0)
    setup_0_02_p_0.inner_forces = np.array([0, 0.2])

    expected_displacement_vector_0_02_p_0 = \
        [0.10204928, 0.16607853, 0.19804137, 0.20984606, 0.21187303,
         0.09148273, -0.06828196, 0.06828192, -0.02335438, 0.02335437,
         -0.00000004, -0.00000003, -0.00000002, -0.00000001, -0.1055135,
         0.10551342, -0.10228252, 0.10228245, -0.09148278, -0.00000004,
         -0.21187311, -0.20984614, -0.19804144, -0.16607859, -0.10204931,
         0.11029064, 0.27796998, 0.48509911, 0.70344497, 0.91900422,
         0.3753411, 0.18050941, 0.1805094, 0.03857951, 0.03857951,
         0.70251881, 0.48159288, 0.27076622, 0.09304104, 0.81112979,
         0.81112978, 0.59214376, 0.59214375, 0.37534111, 0.9188511,
         0.91900423, 0.70344498, 0.48509912, 0.27796999, 0.11029065]

    test_suites.append((setup_0_02_p_0, expected_displacement_vector_0_02_p_0))

    # p = 0

    setup_0_m02_p_0 = DynamicSetup()
    setup_0_m02_p_0.contact_law = make_slope_contact_law(slope=0)
    setup_0_m02_p_0.inner_forces = np.array([0, -0.2])
    expected_displacement_vector_0_m02_p_0 = [-v for v in expected_displacement_vector_0_02_p_0]
    test_suites.append((setup_0_m02_p_0, expected_displacement_vector_0_m02_p_0))

    # various changes

    @dataclass()
    class DynamicSetup(Dynamic):
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

    setup_var = DynamicSetup()
    expected_displacement_vector_var = \
        [-0.00805401, -0.00357142, 0.01047411, 0.02741489, 0.04940722,
         0.06559778, 0.13569877, 0.03575691, 0.04532149, 0.01094824,
         0.18526108, 0.15063882, 0.10631395, 0.05263649, 0.29172965,
         0.11753839, 0.25537485, 0.09294102, 0.20330223, 0.21158742,
         0.40204546, 0.36922423, 0.31434358, 0.24006106, 0.14068317,
         -0.0439969, -0.13577259, -0.26004488, -0.4057771, -0.56621217,
         -0.20248717, -0.10518559, -0.08884555, -0.02061527, -0.01621428,
         -0.42772505, -0.27675558, -0.14872155, -0.04753378, -0.52091162,
         -0.49956963, -0.3589062, -0.34134434, -0.21960863, -0.59684757,
         -0.61573213, -0.44259309, -0.29464628, -0.16950217, -0.0741901]

    test_suites.append((setup_var, expected_displacement_vector_var))

    return test_suites


@pytest.mark.parametrize('setup, expected_displacement_vector', generate_test_suits())
def test_global_optimization_solver(solving_method, setup, expected_displacement_vector):
    runner = DynamicProblem(setup, solving_method)
    results = runner.solve(n_steps=32)
    displacement_vector = results[-1].displacement.T.reshape(1, -1)[0]
    np.set_printoptions(precision=8)
    print(repr(displacement_vector))
    np.testing.assert_array_almost_equal(displacement_vector, expected_displacement_vector, decimal=5)
