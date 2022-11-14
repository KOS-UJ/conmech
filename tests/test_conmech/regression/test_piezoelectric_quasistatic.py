"""
Created at 21.08.2019
"""

from dataclasses import dataclass, field

import numpy as np
import pytest

from conmech.mesh.boundaries_description import BoundariesDescription
from conmech.scenarios.problems import PiezoelectricQuasistaticProblem, PiezoelectricDynamicProblem
from conmech.simulations.problem_solver import PiezoelectricTimeDependentSolver
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
    class QuasistaticSetup(PiezoelectricQuasistaticProblem):
        grid_height: ... = 1
        elements_number: ... = (2, 5)
        mu_coef: ... = 4
        la_coef: ... = 4
        th_coef: ... = 4
        ze_coef: ... = 4
        time_step: ... = 0.1
        contact_law: ... = make_slope_contact_law_piezo(1e1)
        piezoelectricity: ... = field(
            default_factory=lambda: np.array(
                [
                    [[0.0, -0.59, 0.0], [-0.61, 0.0, 0.0], [0.0, 0.0, 0.0]],
                    [[-0.59, 0.0, 0.0], [0.0, 1.14, 0.0], [0.0, 0.0, 0.0]],
                ]
            )
        )
        permittivity: ... = field(
            default_factory=lambda: np.array([[0.1, 0.0, 0.0], [0.0, 0.1, 0.0], [0.0, 0.0, 0.1]])
        )

        @staticmethod
        def inner_forces(x, time=None):
            return np.array([-0.2, -0.2])

        @staticmethod
        def outer_forces(x, time=None):
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
            [0.02739129, 0.01551078],
            [0.04193194, 0.0344827],
            [0.04966183, 0.05129362],
            [0.05372157, 0.06523827],
            [0.05497221, 0.07868415],
            [0.03990124, 0.08298064],
            [0.02610119, 0.08494335],
            [0.02444053, 0.07036843],
            [0.01887287, 0.05538343],
            [0.0105034, 0.03823251],
            [0.00236344, 0.01854232],
            [0.0, 0.0],
            [0.0, 0.0],
        ]
    )
    expected_temperature_vector_m02_m02 = np.asarray(
        [
            -0.17213276,
            -0.08602799,
            -0.0221841,
            0.01236338,
            0.03289128,
            0.04208222,
            0.01226481,
            0.00902652,
            0.02242403,
            0.04431714,
            0.05231734,
            0.03319498,
            -0.0243407,
            -0.06355553,
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

    def inner_forces(x, time=None):
        return np.array([0, 0.2])

    setup_0_02_p_0.inner_forces = inner_forces

    expected_displacement_vector_0_02_p_0 = np.asarray(
        [
            [0.0, 0.0],
            [-0.07690522, -0.06648437],
            [-0.12262936, -0.19051381],
            [-0.1462746, -0.34206085],
            [-0.15508485, -0.50302121],
            [-0.15648751, -0.66223258],
            [0.00000041, -0.66244246],
            [0.15648834, -0.66223257],
            [0.15508565, -0.5030212],
            [0.14627531, -0.34206083],
            [0.12262989, -0.19051379],
            [0.07690553, -0.06648434],
            [0.0, 0.0],
            [0.0, 0.0],
        ]
    )
    expected_temperature_vector_0_02_p_0 = np.asarray(
        [
            0.43120988,
            0.2234753,
            0.05332727,
            -0.0674368,
            -0.1399693,
            -0.1641408,
            -0.16740288,
            -0.16414068,
            -0.13996898,
            -0.06743615,
            0.05332823,
            0.22347662,
            0.43121148,
            0.21898649,
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

    def inner_forces(x, time=None):
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
    class DynamicSetup(PiezoelectricDynamicProblem):
        grid_height: ... = 1.37
        elements_number: ... = (2, 5)
        mu_coef: ... = 4.58
        la_coef: ... = 3.33
        th_coef: ... = 2.11
        ze_coef: ... = 4.99
        time_step: ... = 0.1
        contact_law: ... = make_slope_contact_law_piezo(2.71)
        piezoelectricity: ... = field(
            default_factory=lambda: np.array(
                [
                    [[0.0, -0.59, 0.0], [-0.61, 0.0, 0.0], [0.0, 0.0, 0.0]],
                    [[-0.59, 0.0, 0.0], [0.0, 1.14, 0.0], [0.0, 0.0, 0.0]],
                ]
            )
        )
        permittivity: ... = field(
            default_factory=lambda: np.array([[0.1, 0.0, 0.0], [0.0, 0.1, 0.0], [0.0, 0.0, 0.1]])
        )

        @staticmethod
        def inner_forces(x, time=None):
            return np.array([0, -0.2])

        @staticmethod
        def outer_forces(x, time=None):
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
            [0.02454485, 0.05521995],
            [0.03263034, 0.15296714],
            [0.0326597, 0.28043961],
            [0.02912746, 0.42943956],
            [0.02194101, 0.58807768],
            [-0.13470771, 0.59642012],
            [-0.31508619, 0.59818791],
            [-0.29559302, 0.43025764],
            [-0.25711724, 0.27975335],
            [-0.20030387, 0.14778518],
            [-0.11928462, 0.04488547],
            [0.0, 0.0],
            [0.0, 0.0],
        ]
    )

    expected_temperature_vector_var = np.asarray(
        [
            -0.00912808,
            0.09603883,
            0.14079504,
            0.14273884,
            0.13046765,
            0.15567758,
            -0.02148364,
            -0.17076294,
            -0.1409075,
            -0.14751809,
            -0.17838383,
            -0.2639388,
            -0.39721732,
            -0.00154481,
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
    runner = PiezoelectricTimeDependentSolver(setup, solving_method)
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

    np.testing.assert_array_almost_equal(
        displacement[std_ids], expected_displacement_vector, decimal=3
    )
    precision = 2 if solving_method == "global optimization" else 1  # TODO #94
    np.testing.assert_array_almost_equal(
        electric_potential[std_ids], expected_electric_potential_vector, decimal=precision
    )
