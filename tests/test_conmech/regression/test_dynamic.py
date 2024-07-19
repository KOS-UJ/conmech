"""
Created at 21.08.2019
"""

from dataclasses import dataclass

import numpy as np
import pytest

from conmech.mesh.boundaries_description import BoundariesDescription
from conmech.scenarios.problems import DynamicDisplacementProblem
from conmech.simulations.problem_solver import TimeDependentSolver
from conmech.properties.mesh_description import CrossMeshDescription
from conmech.dynamics.contact.relu_slope_contact_law import make_slope_contact_law
from tests.test_conmech.regression.std_boundary import standard_boundary_nodes


@pytest.fixture(params=["global optimization", "schur"])  # TODO #28
def solving_method(request):
    return request.param


def generate_test_suits():
    test_suites = []

    # Simple example

    @dataclass()
    class DynamicSetup(DynamicDisplacementProblem):
        mu_coef: ... = 4
        la_coef: ... = 4
        th_coef: ... = 4
        ze_coef: ... = 4
        time_step: ... = 0.1
        contact_law: ... = make_slope_contact_law(slope=1e1)

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

    expected_displacement_vector_m02_m02 = [
        [0.0, 0.0],
        [0.03140484, 0.02610888],
        [0.0486117, 0.04198345],
        [0.05738485, 0.05434147],
        [0.06074223, 0.0630033],
        [0.05972214, 0.07028892],
        [0.05171886, 0.0777423],
        [0.04170649, 0.08019597],
        [0.03915195, 0.06898191],
        [0.03129201, 0.05554145],
        [0.01920472, 0.03780141],
        [0.00706545, 0.01626892],
        [0.0, 0.0],
        [0.0, 0.0],
    ]

    test_suites.append((setup_m02_m02, expected_displacement_vector_m02_m02))

    # p = 0 and opposite forces

    setup_0_02_p_0 = DynamicSetup(mesh_descr)
    setup_0_02_p_0.contact_law = make_slope_contact_law(slope=0)

    def inner_forces(x, time=None):
        return np.array([0, 0.2])

    setup_0_02_p_0.inner_forces = inner_forces

    expected_displacement_vector_0_02_p_0 = [
        [0.0, 0.0],
        [-0.09966612, -0.10786224],
        [-0.16211241, -0.27162989],
        [-0.19323258, -0.47377581],
        [-0.20470304, -0.68677563],
        [-0.20666688, -0.8970318],
        [0.00000059, -0.89688304],
        [0.20666805, -0.8970318],
        [0.20470415, -0.68677566],
        [0.19323354, -0.47377587],
        [0.16211313, -0.27162999],
        [0.09966652, -0.10786237],
        [0.0, 0.0],
        [0.0, 0.0],
    ]

    test_suites.append((setup_0_02_p_0, expected_displacement_vector_0_02_p_0))

    # p = 0

    setup_0_m02_p_0 = DynamicSetup(mesh_descr)
    setup_0_m02_p_0.contact_law = make_slope_contact_law(slope=0)

    def inner_forces(x, time=None):
        return np.array([0, -0.2])

    setup_0_m02_p_0.inner_forces = inner_forces

    expected_displacement_vector_0_m02_p_0 = [
        [-v[0], -v[1]] for v in expected_displacement_vector_0_02_p_0
    ]

    test_suites.append((setup_0_m02_p_0, expected_displacement_vector_0_m02_p_0))

    # various changes

    @dataclass()
    class DynamicSetup(DynamicDisplacementProblem):
        mu_coef: ... = 4.58
        la_coef: ... = 3.33
        th_coef: ... = 2.11
        ze_coef: ... = 4.99
        time_step: ... = 0.1
        contact_law: ... = make_slope_contact_law(slope=2.71)

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
    expected_displacement_vector_var = [
        [0.0, 0.0],
        [0.00795525, 0.04239592],
        [0.00329271, 0.13442716],
        [-0.01108608, 0.25795936],
        [-0.02829534, 0.40260094],
        [-0.05043874, 0.56179397],
        [-0.21130529, 0.59231262],
        [-0.40020108, 0.61122983],
        [-0.36747531, 0.43974694],
        [-0.31286012, 0.29330865],
        [-0.23892132, 0.16925194],
        [-0.139945, 0.07466286],
        [0.0, 0.0],
        [0.0, 0.0],
    ]

    test_suites.append((setup_var, expected_displacement_vector_var))

    return test_suites


@pytest.mark.parametrize("setup, expected_displacement_vector", generate_test_suits())
def test_time_dependent_solver(solving_method, setup, expected_displacement_vector):
    runner = TimeDependentSolver(setup, solving_method)
    results = runner.solve(
        n_steps=32,
        initial_displacement=setup.initial_displacement,
        initial_velocity=setup.initial_velocity,
    )

    displacement = results[-1].body.mesh.nodes[:] - results[-1].displaced_nodes[:]
    std_ids = standard_boundary_nodes(runner.body.mesh.nodes, runner.body.mesh.elements)

    # print result
    np.set_printoptions(precision=8, suppress=True)
    print(repr(displacement[std_ids]))

    np.testing.assert_array_almost_equal(
        displacement[std_ids], expected_displacement_vector, decimal=3
    )
