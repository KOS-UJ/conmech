"""
Created at 21.08.2019
"""

from dataclasses import dataclass, field

import numpy as np
import pytest

from conmech.dynamics.contact.contact_law import PotentialOfContactLaw
from conmech.mesh.boundaries_description import BoundariesDescription
from conmech.scenarios.problems import (
    PiezoelectricQuasistaticProblem,
)
from conmech.simulations.problem_solver import PiezoelectricTimeDependentSolver
from conmech.properties.mesh_description import CrossMeshDescription
from conmech.dynamics.contact.p_slope_contact_law import make_slope_contact_law
from tests.test_conmech.regression.std_boundary import standard_boundary_nodes


@pytest.fixture(params=["global optimization", "schur"])  # TODO #28
def solving_method(request):
    return request.param


class PPSlopeContactLaw(PotentialOfContactLaw):
    @staticmethod
    def tangential_bound(
            var_nu: float,
            static_displacement_nu: float,
            dt: float
    ) -> float:
        return - 1.0

    @staticmethod
    def potential_normal_direction(
            var_nu: float,
            static_displacement_nu: float,
            dt: float
    ) -> float:
        """
        electric charge flux

        var_nu == charge
        """
        return 0.0

    @staticmethod
    def potential_tangential_direction(
            var_tau: float,
            static_displacement_tau: float,
            dt: float
    ) -> float:
        """electric charge tangential"""
        return 0.1 * 0.5 * np.linalg.norm(var_tau) ** 2



def generate_test_suits():
    test_suites = []

    # Simple example

    @dataclass()
    class QuasistaticSetup_1(PiezoelectricQuasistaticProblem):
        mu_coef: ... = 4
        la_coef: ... = 4
        th_coef: ... = 4
        ze_coef: ... = 4
        time_step: ... = 0.1
        contact_law: ... = make_slope_contact_law(1e1)
        contact_law_2: ... = PPSlopeContactLaw()
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

    mesh_descr = CrossMeshDescription(
        initial_position=None, max_element_perimeter=0.5, scale=[2.5, 1]
    )
    setup_m02_m02 = QuasistaticSetup_1(mesh_descr)

    expected_displacement_vector_m02_m02 = np.asarray(
        [
            [0., 0.],
            [0.01284572, 0.00727844],
            [0.01998403, 0.01155948],
            [0.02406325, 0.01469925],
            [0.0261574, 0.01657194],
            [0.02644623, 0.01809549],
            [0.02433448, 0.02099327],
            [0.02235926, 0.02215269],
            [0.02133084, 0.01972781],
            [0.01788636, 0.01699742],
            [0.01220893, 0.01293959],
            [0.00540516, 0.00662938],
            [0., 0.],
            [0., 0.],
        ]
    )
    expected_temperature_vector_m02_m02 = np.asarray(
        [-0.09317176, -0.05082419, -0.01792647, 0.00284861, 0.01673696,
         0.02415654, -0.00054601, -0.00339339, 0.00680334, 0.02489092,
         0.04042651, 0.04127354, 0.01658142, -0.02771255,]
    )

    test_suites.append(
        (
            setup_m02_m02,
            expected_displacement_vector_m02_m02,
            expected_temperature_vector_m02_m02,
        )
    )

    # p = 0 and opposite forces

    setup_0_02_p_0 = QuasistaticSetup_1(mesh_descr)
    setup_0_02_p_0.contact_law = make_slope_contact_law(0)

    def inner_forces(x, time=None):
        return np.array([0, 0.2])

    setup_0_02_p_0.inner_forces = inner_forces

    expected_displacement_vector_0_02_p_0 = np.asarray(
        [
            [0., 0.],
            [-0.05119819, -0.04903926],
            [-0.08203236, -0.13236709],
            [-0.09784363, -0.23465962],
            [-0.10375805, -0.34281109],
            [-0.10470465, -0.44949937],
            [0.00000025, -0.44956951],
            [0.10470515, -0.44949937],
            [0.10375853, -0.34281111],
            [0.09784406, -0.23465962],
            [0.08203271, -0.13236709],
            [0.05119838, -0.04903929],
            [0., 0.],
            [0., 0.],
        ]
    )
    expected_temperature_vector_0_02_p_0 = np.asarray(
        [0.32217749, 0.17962827, 0.04606892, -0.04874313, -0.10557165,
         -0.12452466, -0.12761607, -0.12452451, -0.10557137, -0.04874258,
         0.04606974, 0.17962935, 0.32217874, 0.16140012,]
    )

    test_suites.append(
        (
            setup_0_02_p_0,
            expected_displacement_vector_0_02_p_0,
            expected_temperature_vector_0_02_p_0,
        )
    )

    # p = 0

    setup_0_m02_p_0 = QuasistaticSetup_1(mesh_descr)
    setup_0_m02_p_0.contact_law = make_slope_contact_law(0)

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
    class QuasistaticSetup_2(PiezoelectricQuasistaticProblem):
        mu_coef: ... = 4.58
        la_coef: ... = 3.33
        th_coef: ... = 2.11
        ze_coef: ... = 4.99
        time_step: ... = 0.1
        contact_law: ... = make_slope_contact_law(2.71)
        contact_law_2: ... = PPSlopeContactLaw
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

    mesh_descr = CrossMeshDescription(
        initial_position=None, max_element_perimeter=0.685, scale=[3.425, 1.37]
    )
    setup_var = QuasistaticSetup_2(mesh_descr)
    expected_displacement_vector_var = np.asarray(
        [
            [0., 0.],
            [-0.00767607, 0.01454776],
            [-0.0191455, 0.04235841],
            [-0.0297773, 0.07752592],
            [-0.03753689, 0.12139559],
            [-0.0456317, 0.17298394],
            [-0.09733228, 0.18388321],
            [-0.16801203, 0.18878844],
            [-0.15417002, 0.1300507],
            [-0.13068551, 0.08537821],
            [-0.10023372, 0.04869301],
            [-0.05979862, 0.02043574],
            [0., 0.],
            [0., 0.],
        ]
    )

    expected_temperature_vector_var = np.asarray(
        [0.18017535, 0.20951466, 0.19322749, 0.13329264, 0.08185755,
         0.09077227, -0.10696354, -0.26170111, -0.21741867, -0.17859079,
         -0.14847378, -0.1462184, -0.17605883, 0.08383385,]
    )

    test_suites.append(
        (setup_var, expected_displacement_vector_var, expected_temperature_vector_var)
    )

    return test_suites


@pytest.mark.parametrize(
    "setup, expected_displacement_vector, expected_electric_potential_vector",
    generate_test_suits(),
)
def test_piezoelectric_time_dependent_solver(
    solving_method,
    setup,
    expected_displacement_vector,
    expected_electric_potential_vector,
):
    runner = PiezoelectricTimeDependentSolver(setup, solving_method)
    results = runner.solve(
        n_steps=8,
        initial_displacement=setup.initial_displacement,
        initial_velocity=setup.initial_velocity,
        initial_electric_potential=setup.initial_electric_potential,
    )

    std_ids = standard_boundary_nodes(runner.body.mesh.nodes, runner.body.mesh.elements)
    displacement = results[-1].body.mesh.nodes[:] - results[-1].displaced_nodes[:]
    electric_potential = np.zeros(len(results[-1].body.mesh.nodes))
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
        electric_potential[std_ids],
        expected_electric_potential_vector,
        decimal=precision,
    )
