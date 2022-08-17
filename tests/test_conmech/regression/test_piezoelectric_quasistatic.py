"""
Created at 21.08.2019
"""

from dataclasses import dataclass

import numpy as np
import pytest

from conmech.mesh.boundaries_description import BoundariesDescription
from conmech.scenarios.problems import PiezoelectricQuasistatic
from conmech.simulations.problem_solver import (
    PiezoelectricTimeDependent as PiezoelectricQuasistaticSolver,
)
from examples.p_slope_contact_law import make_slope_contact_law
from tests.test_conmech.regression.std_boundary import standard_boundary_nodes


@pytest.fixture(params=["global optimization", "schur"])  # TODO #28
def solving_method(request):
    return request.param


def make_slope_contact_law_piezo(slope):
    class PPSlopeContactLaw(make_slope_contact_law(slope=slope)):
        @staticmethod
        def h_nu(uN, t):
            g_t = 10.7 + t * 0.02
            if uN > g_t:
                return 100.0 * (uN - g_t)
            return 0

        @staticmethod
        def h_tau(uN, t):
            g_t = 10.7 + t * 0.02
            if uN > g_t:
                return 10.0 * (uN - g_t)
            return 0

        @staticmethod
        def h_temp(u_tau):  # potential  # TODO # 48
            return 0.1 * 0.5 * u_tau**2

    return PPSlopeContactLaw


def generate_test_suits():
    test_suites = []

    # Simple example

    @dataclass()
    class QuasistaticSetup(PiezoelectricQuasistatic):
        grid_height: ... = 1
        elements_number: ... = (2, 5)
        mu_coef: ... = 4
        la_coef: ... = 4
        th_coef: ... = 4
        ze_coef: ... = 4
        time_step: ... = 0.1
        contact_law: ... = make_slope_contact_law_piezo(1e1)
        piezoelectricity: ... = np.array([[0.5, 0.0, 0.0], [0.0, 0.5, 0.0], [0.0, 0.0, 0.5]])
        permittivity: ... = np.array([[0.1, 0.0, 0.0], [0.0, 0.1, 0.0], [0.0, 0.0, 0.1]])

        @staticmethod
        def inner_forces(x):
            return np.array([-0.2, -0.2])

        @staticmethod
        def outer_forces(x):
            return np.array([0, 0])

        @staticmethod
        def friction_bound(u_nu):
            return 0

        boundaries: ... = BoundariesDescription(
            contact=lambda x: x[1] == 0, dirichlet=lambda x: x[0] == 0
        )

    setup_m02_m02 = QuasistaticSetup(mesh_type="cross")

    expected_displacement_vector_m02_m02 = np.asarray(
        [
            [0.0, 0.0],
            [0.01203019, 0.01187898],
            [0.01917432, 0.02590137],
            [0.0227568, 0.0416372],
            [0.02414499, 0.05728179],
            [0.02421178, 0.07259322],
            [0.00888861, 0.07357047],
            [-0.0068321, 0.07390116],
            [-0.00714385, 0.05781194],
            [-0.00775304, 0.04121113],
            [-0.00775289, 0.02454146],
            [-0.00555927, 0.00981334],
            [0.0, 0.0],
            [0.0, 0.0],
        ]
    )
    expected_temperature_vector_m02_m02 = np.asarray(
        [
            0.0,
            -0.63441937,
            -1.17577697,
            -1.66472778,
            -2.09002934,
            -2.46838571,
            -2.12202318,
            -1.743185,
            -1.33293631,
            -0.89572128,
            -0.46393988,
            -0.12768763,
            0.0,
            0.0,
        ]
    )

    test_suites.append(
        (
            setup_m02_m02,
            expected_displacement_vector_m02_m02,
            expected_temperature_vector_m02_m02,
        )
    )

    # p = 0 and opposite forces

    setup_0_02_p_0 = QuasistaticSetup(mesh_type="cross")
    setup_0_02_p_0.contact_law = make_slope_contact_law_piezo(0)

    def inner_forces(x):
        return np.array([0, 0.2])

    setup_0_02_p_0.inner_forces = inner_forces

    expected_displacement_vector_0_02_p_0 = np.asarray(
        [
            [0.0, 0.0],
            [-0.01893065, -0.02010763],
            [-0.0310306, -0.05119585],
            [-0.03721498, -0.08999019],
            [-0.03956558, -0.13114238],
            [-0.039986, -0.17184304],
            [-0.00000001, -0.17181239],
            [0.03998598, -0.17184304],
            [0.03956557, -0.13114237],
            [0.03721497, -0.08999019],
            [0.0310306, -0.05119584],
            [0.01893064, -0.02010763],
            [0.0, 0.0],
            [0.0, 0.0],
        ]
    )
    expected_temperature_vector_0_02_p_0 = np.asarray(
        [
            0.0,
            1.2153244,
            2.55984177,
            3.96010252,
            5.3144144,
            6.59457999,
            5.34879737,
            4.10492327,
            2.85093359,
            1.64297766,
            0.62777635,
            0.03664133,
            0.0,
            0.0,
        ]
    )

    test_suites.append(
        (
            setup_0_02_p_0,
            expected_displacement_vector_0_02_p_0,
            expected_temperature_vector_0_02_p_0,
        )
    )

    # p = 0

    setup_0_m02_p_0 = QuasistaticSetup(mesh_type="cross")
    setup_0_m02_p_0.contact_law = make_slope_contact_law_piezo(0)

    def inner_forces(x):
        return np.array([0, -0.2])

    setup_0_m02_p_0.inner_forces = inner_forces
    expected_displacement_vector_0_m02_p_0 = [-v for v in expected_displacement_vector_0_02_p_0]
    expected_temperature_vector_0_m02_p_0 = [-v for v in expected_temperature_vector_0_02_p_0]
    test_suites.append(
        (
            setup_0_m02_p_0,
            expected_displacement_vector_0_m02_p_0,
            expected_temperature_vector_0_m02_p_0,
        )
    )

    # various changes

    @dataclass()
    class QuasistaticSetup(PiezoelectricQuasistatic):
        grid_height: ... = 1.37
        elements_number: ... = (2, 5)
        mu_coef: ... = 4.58
        la_coef: ... = 3.33
        th_coef: ... = 2.11
        ze_coef: ... = 4.99
        time_step: ... = 0.1
        contact_law: ... = make_slope_contact_law_piezo(2.71)
        piezoelectricity: ... = np.array([[0.5, 0.0, 0.0], [0.0, 0.5, 0.0], [0.0, 0.0, 0.5]])
        permittivity: ... = np.array([[0.1, 0.0, 0.0], [0.0, 0.1, 0.0], [0.0, 0.0, 0.1]])

        @staticmethod
        def inner_forces(x):
            return np.array([0, -0.2])

        @staticmethod
        def outer_forces(x):
            return np.array([0.3, 0.0])

        @staticmethod
        def friction_bound(u_nu):
            return 0.0

        boundaries: ... = BoundariesDescription(
            contact=lambda x: x[1] == 0, dirichlet=lambda x: x[0] == 0
        )

    setup_var = QuasistaticSetup(mesh_type="cross")
    expected_displacement_vector_var = np.asarray(
        [
            [0.0, 0.0],
            [0.03437438, 0.03619713],
            [0.05744414, 0.10252049],
            [0.06985391, 0.19077923],
            [0.07445972, 0.28934762],
            [0.07427655, 0.3896374],
            [-0.02460142, 0.3901174],
            [-0.12651214, 0.39110595],
            [-0.12140669, 0.28981308],
            [-0.10902829, 0.19179337],
            [-0.08653477, 0.10403044],
            [-0.05102786, 0.03755931],
            [0.0, 0.0],
            [0.0, 0.0],
        ]
    )

    expected_temperature_vector_var = np.asarray(
        [
            0.0,
            -1.71049187,
            -3.92294138,
            -6.41571825,
            -8.98860905,
            -11.50379181,
            -9.0379844,
            -6.42830647,
            -3.98800438,
            -1.87728268,
            -0.30094523,
            0.39254945,
            0.0,
            0.0,
        ]
    )

    test_suites.append(
        (setup_var, expected_displacement_vector_var, expected_temperature_vector_var)
    )

    return test_suites


@pytest.mark.parametrize(
    "setup, expected_displacement_vector, expected_electric_potential_vector",
    generate_test_suits(),
)
def test_global_optimization_solver(
    solving_method, setup, expected_displacement_vector, expected_electric_potential_vector
):
    runner = PiezoelectricQuasistaticSolver(setup, solving_method)
    results = runner.solve(
        n_steps=32,
        initial_displacement=setup.initial_displacement,
        initial_velocity=setup.initial_velocity,
        initial_electric_potential=setup.initial_electric_potential,
    )

    std_ids = standard_boundary_nodes(runner.body.mesh.initial_nodes, runner.body.mesh.elements)
    displacement = results[-1].body.mesh.initial_nodes[:] - results[-1].displaced_nodes[:]
    electric_potential = np.zeros(len(results[-1].body.mesh.initial_nodes))
    electric_potential[: len(results[-1].electric_potential)] = results[-1].electric_potential

    # print result
    np.set_printoptions(precision=8, suppress=True)
    print(repr(displacement[std_ids]))
    print(repr(electric_potential[std_ids]))

    precision = 2 if solving_method == "global optimization" else 3
    np.testing.assert_array_almost_equal(
        displacement[std_ids], expected_displacement_vector, decimal=3
    )
    np.testing.assert_array_almost_equal(
        electric_potential[std_ids], expected_electric_potential_vector, decimal=precision
    )
