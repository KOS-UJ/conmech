"""
Created at 21.08.2019
"""

from dataclasses import dataclass

import numpy as np
import pytest

from conmech.mesh.boundaries_description import BoundariesDescription
from conmech.scenarios.problems import QuasistaticDisplacementProblem
from conmech.simulations.problem_solver import TimeDependentSolver
from examples.p_slope_contact_law import make_slope_contact_law
from tests.test_conmech.regression.std_boundary import standard_boundary_nodes


@pytest.fixture(params=["global optimization", "schur"])  # TODO #28
def solving_method(request):
    return request.param


def generate_test_suits():
    test_suites = []

    # Simple example

    @dataclass()
    class QuasistaticSetup(QuasistaticDisplacementProblem):
        grid_height: ... = 1
        elements_number: ... = (2, 5)
        mu_coef: ... = 4
        la_coef: ... = 4
        th_coef: ... = 4
        ze_coef: ... = 4
        time_step: ... = 0.1
        contact_law: ... = make_slope_contact_law(slope=1e1)

        @staticmethod
        def inner_forces(x, v=None, time=None):
            return np.array([-0.2, -0.2])

        @staticmethod
        def outer_forces(x, v=None, time=None):
            return np.array([0, 0])

        @staticmethod
        def friction_bound(u_nu):
            return 0

        boundaries: ... = BoundariesDescription(
            contact=lambda x: x[1] == 0, dirichlet=lambda x: x[0] == 0
        )

    setup_m02_m02 = QuasistaticSetup(mesh_type="cross")

    expected_displacement_vector_m02_m02 = [
        [0.0, 0.0],
        [0.03133183, 0.02696152],
        [0.04827075, 0.04292597],
        [0.05681381, 0.05580324],
        [0.06001541, 0.06502902],
        [0.0589324, 0.07290283],
        [0.05036095, 0.08038138],
        [0.0398147, 0.08284383],
        [0.03734164, 0.0711117],
        [0.0297232, 0.05717822],
        [0.0179927, 0.03898402],
        [0.00632795, 0.01697416],
        [0.0, 0.0],
        [0.0, 0.0],
    ]

    test_suites.append((setup_m02_m02, expected_displacement_vector_m02_m02))

    # p = 0 and opposite forces

    setup_0_02_p_0 = QuasistaticSetup(mesh_type="cross")
    setup_0_02_p_0.contact_law = make_slope_contact_law(slope=0)

    def inner_forces(x, v=None, time=None):
        return np.array([0, 0.2])

    setup_0_02_p_0.inner_forces = inner_forces

    expected_displacement_vector_0_02_p_0 = [
        [0.0, 0.0],
        [-0.11229405, -0.11927587],
        [-0.18406934, -0.30368731],
        [-0.22075417, -0.53381063],
        [-0.23469758, -0.7779203],
        [-0.23719137, -1.01935163],
        [0.00000057, -1.01916981],
        [0.23719248, -1.01935164],
        [0.23469865, -0.77792033],
        [0.22075512, -0.53381069],
        [0.18407005, -0.3036874],
        [0.11229443, -0.11927601],
        [0.0, 0.0],
        [0.0, 0.0],
    ]

    test_suites.append((setup_0_02_p_0, expected_displacement_vector_0_02_p_0))

    # p = 0

    setup_0_m02_p_0 = QuasistaticSetup(mesh_type="cross")
    setup_0_m02_p_0.contact_law = make_slope_contact_law(slope=0)

    def inner_forces(x, v=None, time=None):
        return np.array([0, -0.2])

    setup_0_m02_p_0.inner_forces = inner_forces

    expected_displacement_vector_0_m02_p_0 = [
        [-v[0], -v[1]] for v in expected_displacement_vector_0_02_p_0
    ]

    test_suites.append((setup_0_m02_p_0, expected_displacement_vector_0_m02_p_0))

    # various changes

    @dataclass()
    class QuasistaticSetup(QuasistaticDisplacementProblem):
        grid_height: ... = 1.37
        elements_number: ... = (2, 5)
        mu_coef: ... = 4.58
        la_coef: ... = 3.33
        th_coef: ... = 2.11
        ze_coef: ... = 4.99
        time_step: ... = 0.1
        contact_law: ... = make_slope_contact_law(slope=2.71)

        @staticmethod
        def inner_forces(x, v=None, time=None):
            return np.array([0, -0.2])

        @staticmethod
        def outer_forces(x, v=None, time=None):
            return np.array([0.3, 0.0])

        @staticmethod
        def friction_bound(u_nu):
            return 0.0

        boundaries: ... = BoundariesDescription(
            contact=lambda x: x[1] == 0, dirichlet=lambda x: x[0] == 0
        )

    setup_var = QuasistaticSetup(mesh_type="cross")
    expected_displacement_vector_var = [
        [0.0, 0.0],
        [0.0198434, 0.0502673],
        [0.02405844, 0.1543526],
        [0.01578334, 0.29324725],
        [0.00215192, 0.45464833],
        [-0.01836903, 0.63040914],
        [-0.19559099, 0.65949265],
        [-0.40091827, 0.67791889],
        [-0.36882488, 0.48969365],
        [-0.31542913, 0.3265841],
        [-0.24222533, 0.18712542],
        [-0.14246846, 0.08065452],
        [0.0, 0.0],
        [0.0, 0.0],
    ]

    test_suites.append((setup_var, expected_displacement_vector_var))

    return test_suites


@pytest.mark.parametrize("setup, expected_displacement_vector", generate_test_suits())
def test_global_optimization_solver(solving_method, setup, expected_displacement_vector):
    # TODO: #65 Duplicated neumann node  in old boundary construction
    runner = TimeDependentSolver(setup, solving_method)
    results = runner.solve(
        n_steps=32,
        initial_displacement=setup.initial_displacement,
        initial_velocity=setup.initial_velocity,
    )

    displacement = results[-1].body.mesh.initial_nodes[:] - results[-1].displaced_nodes[:]
    std_ids = standard_boundary_nodes(runner.body.mesh.initial_nodes, runner.body.mesh.elements)

    # print result
    np.set_printoptions(precision=8, suppress=True)
    print(repr(displacement[std_ids]))

    np.testing.assert_array_almost_equal(
        displacement[std_ids], expected_displacement_vector, decimal=3
    )
