"""
Created at 21.08.2019
"""

from dataclasses import dataclass

import numpy as np
import pytest

from conmech.mesh.boundaries_description import BoundariesDescription
from conmech.scenarios.problems import TemperatureDynamic
from conmech.simulations.problem_solver import TemperatureTimeDependent as TDynamicProblem
from examples.p_slope_contact_law import make_slope_contact_law
from tests.test_conmech.regression.std_boundary import standard_boundary_nodes


@pytest.fixture(params=["global optimization", "schur"])  # TODO #28
def solving_method(request):
    return request.param


def make_slope_contact_law_temp(slope):
    class TPSlopeContactLaw(make_slope_contact_law(slope=slope)):
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
        def h_temp(vTnorm):
            return 0.1 * vTnorm

    return TPSlopeContactLaw


def generate_test_suits():
    test_suites = []

    # Simple example

    @dataclass()
    class DynamicSetup(TemperatureDynamic):
        grid_height: ... = 1
        elements_number: ... = (2, 5)
        mu_coef: ... = 4
        la_coef: ... = 4
        th_coef: ... = 4
        ze_coef: ... = 4
        time_step: ... = 0.1
        contact_law: ... = make_slope_contact_law_temp(1e1)
        thermal_expansion: ... = np.array([[0.5, 0.0, 0.0], [0.0, 0.5, 0.0], [0.0, 0.0, 0.5]])
        thermal_conductivity: ... = np.array([[0.1, 0.0, 0.0], [0.0, 0.1, 0.0], [0.0, 0.0, 0.1]])

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

    setup_m02_m02 = DynamicSetup(mesh_type="cross")

    expected_displacement_vector_m02_m02 = np.asarray(
        [
            [0., 0.],
            [0.03142428, 0.02691658],
            [0.04839312, 0.04238928],
            [0.05698615, 0.05458973],
            [0.06022096, 0.06307318],
            [0.0591116, 0.07018657],
            [0.05128066, 0.07756148],
            [0.04143363, 0.07994437],
            [0.03895115, 0.06888961],
            [0.03117291, 0.05560756],
            [0.01913418, 0.03800219],
            [0.00700475, 0.01647643],
            [0., 0.],
            [0., 0.],
        ]
    )
    expected_temperature_vector_m02_m02 = np.asarray(
        [0., 0.00735363, 0.00913178, 0.00828201, 0.00694812,
         0.00636948, 0.00607767, 0.00578943, 0.00648897, 0.00789741,
         0.00825697, 0.00542543, 0., 0.]
    )

    test_suites.append(
        (
            setup_m02_m02,
            expected_displacement_vector_m02_m02,
            expected_temperature_vector_m02_m02,
        )
    )

    # p = 0 and opposite forces

    setup_0_02_p_0 = DynamicSetup(mesh_type="cross")
    setup_0_02_p_0.contact_law = make_slope_contact_law_temp(0)

    def inner_forces(x):
        return np.array([0, 0.2])

    setup_0_02_p_0.inner_forces = inner_forces

    expected_displacement_vector_0_02_p_0 = np.asarray(
        [
            [0., 0.],
            [-0.09962716, -0.10791196],
            [-0.16199068, -0.27160136],
            [-0.19306197, -0.47359855],
            [-0.20451057, -0.68641614],
            [-0.20646559, -0.89647558],
            [0.00000059, -0.89632143],
            [0.20646675, -0.89647557],
            [0.20451168, -0.68641617],
            [0.19306293, -0.47359861],
            [0.16199141, -0.27160145],
            [0.09962757, -0.10791209],
            [0., 0.],
            [0., 0.]
        ]
    )
    expected_temperature_vector_0_02_p_0 = np.asarray(
        [0., -0.01368016, -0.01005095, -0.00546105, -0.00217319,
         -0.00098241, -0.00000016, 0.00098209, 0.00217289, 0.0054608,
         0.01005075, 0.01368001, 0., 0.]
    )

    test_suites.append(
        (
            setup_0_02_p_0,
            expected_displacement_vector_0_02_p_0,
            expected_temperature_vector_0_02_p_0,
        )
    )

    # p = 0

    setup_0_m02_p_0 = DynamicSetup(mesh_type="cross")
    setup_0_m02_p_0.contact_law = make_slope_contact_law_temp(0)

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
    class DynamicSetup(TemperatureDynamic):
        grid_height: ... = 1.37
        elements_number: ... = (2, 5)
        mu_coef: ... = 4.58
        la_coef: ... = 3.33
        th_coef: ... = 2.11
        ze_coef: ... = 4.99
        time_step: ... = 0.1
        contact_law: ... = make_slope_contact_law_temp(2.71)
        thermal_expansion: ... = np.array([[0.5, 0.0, 0.0], [0.0, 0.5, 0.0], [0.0, 0.0, 0.5]])
        thermal_conductivity: ... = np.array([[0.1, 0.0, 0.0], [0.0, 0.1, 0.0], [0.0, 0.0, 0.1]])

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

    setup_var = DynamicSetup(mesh_type="cross")
    expected_displacement_vector_var = np.asarray(
        [
            [0., 0.],
            [0.00826122, 0.0421944],
            [0.00410242, 0.13421482],
            [-0.00990836, 0.25783836],
            [-0.02692933, 0.40249172],
            [-0.04900262, 0.56156278],
            [-0.20968181, 0.59213072],
            [-0.39831803, 0.6111903],
            [-0.36578831, 0.43996181],
            [-0.31149975, 0.29375818],
            [-0.23806523, 0.16985506],
            [-0.13971534, 0.07516066],
            [0., 0.],
            [0., 0.],
        ]
    )

    expected_temperature_vector_var = np.asarray(
        [0., -0.00870148, -0.01319496, -0.00947956, -0.00501162,
         -0.00299865, -0.00606658, -0.00913445, -0.01288869, -0.02153761,
         -0.03067854, -0.03005361, 0., 0.]
    )

    test_suites.append(
        (setup_var, expected_displacement_vector_var, expected_temperature_vector_var)
    )

    return test_suites


@pytest.mark.parametrize(
    "setup, expected_displacement_vector, expected_temperature_vector",
    generate_test_suits(),
)
def test_global_optimization_solver(
    solving_method, setup, expected_displacement_vector, expected_temperature_vector
):
    runner = TDynamicProblem(setup, solving_method)
    results = runner.solve(
        n_steps=32,
        initial_displacement=setup.initial_displacement,
        initial_velocity=setup.initial_velocity,
        initial_temperature=setup.initial_temperature,
    )

    std_ids = standard_boundary_nodes(runner.body.mesh.initial_nodes, runner.body.mesh.elements)
    displacement = results[-1].body.mesh.initial_nodes[:] - results[-1].displaced_nodes[:]
    temperature = np.zeros(len(results[-1].body.mesh.initial_nodes))
    temperature[: len(results[-1].temperature)] = results[-1].temperature

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
