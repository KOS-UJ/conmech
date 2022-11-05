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
            [0., 0.],
            [0.02763052, 0.01548599],
            [0.04226993, 0.0346185],
            [0.05004803, 0.05148295],
            [0.05414639, 0.06541078],
            [0.05542924, 0.07884249],
            [0.04033042, 0.08318157],
            [0.02655525, 0.08517122],
            [0.02487058, 0.0706121],
            [0.01921677, 0.05564069],
            [0.01071263, 0.03846692],
            [0.00243661, 0.01870481],
            [0., 0.],
            [0., 0.],
        ]
    )
    expected_temperature_vector_m02_m02 = np.asarray(
        [-0.17255886, -0.08594389, -0.02195304, 0.01251428, 0.03297251,
         0.04210582, 0.01216087, 0.00891185, 0.02240036, 0.04450981,
         0.05255774, 0.03339226, -0.02443637, -0.06380221,]
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
            [0., 0.],
            [-0.07743045, -0.06669382],
            [-0.12348913, -0.19152753],
            [-0.14731263, -0.34406718],
            [-0.15619039, -0.50613139],
            [-0.15760661, -0.66646509],
            [0.00000037, -0.66667919],
            [0.15760738, -0.66646508],
            [0.15619112, -0.50613138],
            [0.14731328, -0.34406717],
            [0.12348961, -0.19152751],
            [0.07743073, -0.06669379],
            [0., 0.],
            [0., 0.],
        ]
    )
    expected_temperature_vector_0_02_p_0 = np.asarray(
        [0.43200774, 0.22364054, 0.05335082, -0.06754468, -0.14015486,
         -0.16435208, -0.16761522, -0.16435188, -0.14015455, -0.06754407,
         0.05335171, 0.22364174, 0.43200924, 0.21949495,]
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
            [0., 0.],
            [0.02473405, 0.0556681],
            [0.03285925, 0.15410985],
            [0.03287611, 0.28249185],
            [0.02930375, 0.43259534],
            [0.02207896, 0.59239179],
            [-0.13574097, 0.60071622],
            [-0.31735658, 0.60242367],
            [-0.29772689, 0.43326179],
            [-0.25897055, 0.28160027],
            [-0.20172852, 0.14860058],
            [-0.12009683, 0.04484386],
            [0., 0.],
            [0., 0.],
        ]
    )

    expected_temperature_vector_var = np.asarray(
        [-0.01001105, 0.09590015, 0.14064874, 0.14283285, 0.1306944,
         0.155932, -0.02129994, -0.17046025, -0.14060958, -0.14718833,
         -0.17799561, -0.2638995, -0.39813996, -0.00213899,]
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
