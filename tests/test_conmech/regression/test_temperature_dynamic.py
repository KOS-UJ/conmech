"""
Created at 21.08.2019
"""

from dataclasses import dataclass, field

import numpy as np
import pytest

from conmech.dynamics.contact.contact_law import PotentialOfContactLaw
from conmech.mesh.boundaries_description import BoundariesDescription
from conmech.scenarios.problems import TemperatureDynamicProblem
from conmech.simulations.problem_solver import TemperatureTimeDependentSolver
from conmech.properties.mesh_description import CrossMeshDescription
from conmech.dynamics.contact.p_slope_contact_law import make_slope_contact_law
from tests.test_conmech.regression.std_boundary import standard_boundary_nodes


@pytest.fixture(params=["global optimization", "schur"])  # TODO #28
def solving_method(request):
    return request.param


class TPSlopeContactLaw(PotentialOfContactLaw):
    @staticmethod
    def normal_bound(
            var_nu: float,
            static_displacement_nu: float,
            dt: float
    ) -> float:
        """
        Direction of heat flux
        """
        return - 1.0

    @staticmethod
    def potential_normal_direction(
            var_nu: float,
            static_displacement_nu: float,
            dt: float
    ) -> float:
        """Temperature exchange"""
        return 0.0

    @staticmethod
    def potential_tangential_direction(
            var_tau: float,
            static_displacement_tau: float,
            dt: float
    ) -> float:
        """Friction generated temperature"""
        return 0.0


def generate_test_suits():
    test_suites = []

    # Simple example

    @dataclass()
    class DynamicSetup(TemperatureDynamicProblem):
        mu_coef: ... = 4
        la_coef: ... = 4
        th_coef: ... = 4
        ze_coef: ... = 4
        time_step: ... = 0.1
        contact_law: ... = make_slope_contact_law(1e1)
        contact_law_2: ... = TPSlopeContactLaw()
        thermal_expansion: ... = field(
            default_factory=lambda: np.array([[0.5, 0.0, 0.0], [0.0, 0.5, 0.0], [0.0, 0.0, 0.5]])
        )
        thermal_conductivity: ... = field(
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
    setup_m02_m02 = DynamicSetup(mesh_descr)

    expected_displacement_vector_m02_m02 = np.asarray(
        [
            [0., 0.],
            [0.00570683, 0.00431404],
            [0.00857057, 0.00526714],
            [0.01005346, 0.00586446],
            [0.01058508, 0.00597765],
            [0.01028078, 0.00577634],
            [0.01051063, 0.0074202],
            [0.01045592, 0.00795605],
            [0.0099967, 0.00760991],
            [0.00853351, 0.00684343],
            [0.00600395, 0.00531373],
            [0.00282262, 0.00263098],
            [0., 0.],
            [0., 0.],
        ]
    )
    expected_temperature_vector_m02_m02 = np.asarray(
        [
            0., 0.00407167, 0.00284784, 0.00232436, 0.00175827,
            0.0014074, 0.00094456, 0.00049082, 0.00088726, 0.00154845,
            0.00210648, 0.00257013, 0., 0.,
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

    setup_0_02_p_0 = DynamicSetup(mesh_descr)
    setup_0_02_p_0.contact_law = make_slope_contact_law(0)

    def inner_forces(x, time=None):
        return np.array([0, 0.2])

    setup_0_02_p_0.inner_forces = inner_forces

    expected_displacement_vector_0_02_p_0 = np.asarray(
        [
            [0., 0.],
            [-0.00357335, -0.00721697],
            [-0.00400228, -0.01306104],
            [-0.00328983, -0.01728046],
            [-0.00266257, -0.02009906],
            [-0.00246619, -0.02248587],
            [0.00000012, -0.02249418],
            [0.00246643, -0.02248586],
            [0.00266281, -0.02009906],
            [0.00329004, -0.01728048],
            [0.00400244, -0.01306106],
            [0.00357344, -0.007217],
            [0., 0.],
            [0., 0.],
        ]
    )
    expected_temperature_vector_0_02_p_0 = np.asarray(
        [
            0.0, -0.00125683, 0.00007371, 0.00030325, 0.00019259,
            0.00008153, -0.00000002, -0.00008157, -0.00019262, -0.00030328,
            -0.00007372, 0.00125684, 0.0, 0.0,
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

    setup_0_m02_p_0 = DynamicSetup(mesh_descr)
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
    class DynamicSetup(TemperatureDynamicProblem):
        mu_coef: ... = 4.58
        la_coef: ... = 3.33
        th_coef: ... = 2.11
        ze_coef: ... = 4.99
        time_step: ... = 0.1
        contact_law: ... = make_slope_contact_law(2.71)
        contact_law_2: ... = TPSlopeContactLaw
        thermal_expansion: ... = field(
            default_factory=lambda: np.array([[0.5, 0.0, 0.0], [0.0, 0.5, 0.0], [0.0, 0.0, 0.5]])
        )
        thermal_conductivity: ... = field(
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
    setup_var = DynamicSetup(mesh_descr)
    expected_displacement_vector_var = np.asarray(
        [
            [0., 0.],
            [-0.00224574, 0.00303146],
            [-0.00635254, 0.00724873],
            [-0.01133355, 0.01098753],
            [-0.01611561, 0.01621815],
            [-0.02404372, 0.02594285],
            [-0.03629391, 0.03524607],
            [-0.06458808, 0.04365407],
            [-0.05360272, 0.02541212],
            [-0.04325996, 0.01882516],
            [-0.0340562, 0.01503337],
            [-0.0222147, 0.01126095],
            [0., 0.],
            [0., 0.],
        ]
    )

    expected_temperature_vector_var = np.asarray(
        [0., -0.00160508, -0.00133705, -0.00092485, -0.00083582,
         -0.00073482, -0.00167299, -0.00278935, -0.00325494, -0.00339006,
         -0.00344784, -0.00747268, 0., 0.,]
    )

    test_suites.append(
        (setup_var, expected_displacement_vector_var, expected_temperature_vector_var)
    )

    return test_suites


@pytest.mark.parametrize(
    "setup, expected_displacement_vector, expected_temperature_vector",
    generate_test_suits(),
)
def test_temperature_time_dependent_solver(
    solving_method, setup, expected_displacement_vector, expected_temperature_vector
):
    runner = TemperatureTimeDependentSolver(setup, solving_method)
    result_generator = runner.solve(
        n_steps=4,
        initial_displacement=setup.initial_displacement,
        initial_velocity=setup.initial_velocity,
        initial_temperature=setup.initial_temperature,
    )
    # replace generator with collection
    results = tuple(result_generator)

    std_ids = standard_boundary_nodes(runner.body.mesh.nodes, runner.body.mesh.elements)
    displacement = results[-1].body.mesh.nodes[:] - results[-1].displaced_nodes[:]
    temperature = np.zeros(len(results[-1].body.mesh.nodes))
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
