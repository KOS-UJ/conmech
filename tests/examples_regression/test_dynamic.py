"""
Created at 21.08.2019
"""

import numpy as np
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
        [-0.03177345, -0.04913543, -0.05796524, -0.06133454, -0.06028476,
         -0.04819916, -0.0210363, -0.03510663, -0.0066299, -0.01250094,
         -0.051819, -0.0462789, -0.03564031, -0.01921912, -0.04751813,
         -0.05726999, -0.04333058, -0.05521844, -0.03419117, -0.05250087,
         -0.0426713, -0.04006988, -0.03205238, -0.01970323, -0.00727496,
         -0.02647471, -0.04241128, -0.05465171, -0.06308749, -0.07012755,
         -0.04972115, -0.0291698, -0.033125, -0.00823664, -0.01066615,
         -0.0685363, -0.05728541, -0.0416113, -0.02044928, -0.07465047,
         -0.07058359, -0.06352885, -0.06166474, -0.04894518, -0.0777147,
         -0.08021195, -0.06917232, -0.0558679, -0.03814094, -0.0164544]

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
        [-0.00786375, -0.00356065, 0.01038378, 0.02727653, 0.04925508,
         0.06564938, 0.13580744, 0.03596722, 0.04530518, 0.01093658,
         0.1853893, 0.15077572, 0.10649803, 0.05271624, 0.29199804,
         0.11752764, 0.25563051, 0.09294711, 0.20352497, 0.21171391,
         0.40245773, 0.36962829, 0.31469921, 0.24023932, 0.14062184,
         -0.04253131, -0.13504396, -0.25953535, -0.40555009, -0.56626764,
         -0.20181801, -0.10438878, -0.08784181, -0.02026451, -0.01565589,
         -0.42747608, -0.27619964, -0.14786973, -0.04671108, -0.52081014,
         -0.49947544, -0.35849417, -0.34095378, -0.21888928, -0.596896,
         -0.61577913, -0.44234434, -0.29408182, -0.16871166, -0.07353924]

    test_suites.append((setup_var, expected_displacement_vector_var))

    return test_suites


@pytest.mark.parametrize('setup, expected_displacement_vector', generate_test_suits())
def test_global_optimization_solver(solving_method, setup, expected_displacement_vector):
    runner = DynamicProblem(setup, solving_method)
    results = runner.solve(n_steps=32)
    displacement_vector = results[-1].displacement.T.reshape(1, -1)[0]
    np.testing.assert_array_almost_equal(displacement_vector, expected_displacement_vector, decimal=5)
