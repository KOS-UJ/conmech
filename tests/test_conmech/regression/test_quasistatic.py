"""
Created at 21.08.2019
"""

from dataclasses import dataclass

import numpy as np
import pytest

from conmech.mesh.boundaries_description import BoundariesDescription
from conmech.scenarios.problems import QuasistaticDisplacementProblem
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
    class QuasistaticSetup(QuasistaticDisplacementProblem):
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

        boundaries: ... = BoundariesDescription(
            contact=lambda x: x[1] == 0, dirichlet=lambda x: x[0] == 0
        )

    mesh_descr = CrossMeshDescription(
        initial_position=None, max_element_perimeter=0.5, scale=[2.5, 1]
    )
    setup_m02_m02 = QuasistaticSetup(mesh_descr)

    # setup0
    expected_displacement_vector_m02_m02 = [
        [0.0, 0.0],
        [0.03110494, 0.02605975],
        [0.0480314, 0.04222483],
        [0.05652026, 0.05498411],
        [0.05961243, 0.064079],
        [0.0584169, 0.07115007],
        [0.05034688, 0.07886201],
        [0.04010372, 0.08137039],
        [0.03759424, 0.06998544],
        [0.02988933, 0.0563158],
        [0.01813132, 0.03828731],
        [0.00648342, 0.01652978],
        [0.0, 0.0],
        [0.0, 0.0],
    ]

    test_suites.append((setup_m02_m02, expected_displacement_vector_m02_m02))

    # p = 0 and opposite forces

    setup_0_02_p_0 = QuasistaticSetup(mesh_descr)
    setup_0_02_p_0.contact_law = make_slope_contact_law(slope=0)

    def inner_forces(x, time=None):
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

    setup_0_m02_p_0 = QuasistaticSetup(mesh_descr)
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
    class QuasistaticSetup(QuasistaticDisplacementProblem):
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

        boundaries: ... = BoundariesDescription(
            contact=lambda x: x[1] == 0, dirichlet=lambda x: x[0] == 0
        )

    mesh_descr = CrossMeshDescription(
        initial_position=None, max_element_perimeter=0.685, scale=[3.425, 1.37]
    )
    setup_var = QuasistaticSetup(mesh_descr)
    # setup3
    expected_displacement_vector_var = [
        [0.0, 0.0],
        [0.01884863, 0.0504626],
        [0.02148185, 0.15327319],
        [0.01145928, 0.28855596],
        [-0.0037091, 0.4443636],
        [-0.02518466, 0.61085737],
        [-0.19518967, 0.6418065],
        [-0.39461876, 0.66067796],
        [-0.36292907, 0.4787971],
        [-0.31087128, 0.32140352],
        [-0.23950126, 0.18562144],
        [-0.14140114, 0.08064638],
        [0.0, 0.0],
        [0.0, 0.0],
    ]

    test_suites.append((setup_var, expected_displacement_vector_var))

    return test_suites


@pytest.mark.parametrize("setup, expected_displacement_vector", generate_test_suits())
def test_time_dependent_solver(solving_method, setup, expected_displacement_vector):
    # TODO: #65 Duplicated neumann node  in old boundary construction
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
