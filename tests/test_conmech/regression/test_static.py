"""
Created at 21.08.2019
"""

from dataclasses import dataclass

import numpy as np
import pytest

from conmech.mesh.boundaries_description import BoundariesDescription
from conmech.scenarios.problems import StaticDisplacementProblem
from conmech.simulations.problem_solver import StaticSolver
from conmech.properties.mesh_description import CrossMeshDescription
from conmech.dynamics.contact.relu_slope_contact_law import make_slope_contact_law
from tests.test_conmech.regression.std_boundary import standard_boundary_nodes

try:
    import kosopt

    available_opt_mtds = ["BFGS", "qsm"]
except ImportError:
    available_opt_mtds = ["BFGS"]


@pytest.fixture(params=["direct", "global optimization", "schur"])
def solving_method(request):
    return request.param


@pytest.fixture(params=available_opt_mtds)
def opt_method(request):
    return request.param


def generate_test_suits():
    test_suites = []

    # Simple example

    @dataclass()
    class StaticSetup(StaticDisplacementProblem):
        mu_coef: ... = 4
        la_coef: ... = 4
        contact_law: ... = make_slope_contact_law(slope=1)

        @staticmethod
        def inner_forces(x, t=None):
            return np.array([-0.2, -0.2])

        @staticmethod
        def outer_forces(x, t=None):
            return np.array([0, 0])

        boundaries: ... = BoundariesDescription(
            contact=lambda x: x[1] == 0, dirichlet=lambda x: x[0] == 0
        )

    mesh_descr = CrossMeshDescription(
        initial_position=None, max_element_perimeter=0.5, scale=[2.5, 1]
    )
    setup_m02_m02 = StaticSetup(mesh_descr)

    # setup0
    expected_displacement_vector_m02_m02 = [
        [0., 0.],
        [0.04843765, 0.04432924],
        [0.0764282, 0.08752198],
        [0.09064934, 0.13303217],
        [0.09615089, 0.17621039],
        [0.09566344, 0.21773051],
        [0.05373814, 0.22481468],
        [0.00963019, 0.22717042],
        [0.00731451, 0.18138176],
        [0.0010484, 0.13299365],
        [-0.00636478, 0.08194551],
        [-0.00867577, 0.03339803],
        [0., 0.],
        [0., 0.],
    ]

    test_suites.append((setup_m02_m02, expected_displacement_vector_m02_m02))

    # p = 0 and opposite forces

    setup_0_02_p_0 = StaticSetup(mesh_descr)
    setup_0_02_p_0.contact_law = make_slope_contact_law(slope=0)

    def inner_forces(x, t=None):
        return np.array([0, 0.2])

    setup_0_02_p_0.inner_forces = inner_forces

    # setup1
    expected_displacement_vector_0_02_p_0 = [
        [0.0, 0.0],
        [-0.11787841, -0.1252073],
        [-0.19322311, -0.31878912],
        [-0.23173236, -0.56035602],
        [-0.24636923, -0.8166048],
        [-0.24898703, -1.07004206],
        [0.0, -1.06985119],
        [0.24898703, -1.07004206],
        [0.24636923, -0.8166048],
        [0.23173236, -0.56035602],
        [0.19322311, -0.31878912],
        [0.11787841, -0.1252073],
        [0.0, 0.0],
        [0.0, 0.0],
    ]

    test_suites.append((setup_0_02_p_0, expected_displacement_vector_0_02_p_0))

    # p = 0

    setup_0_m02_p_0 = StaticSetup(mesh_descr)
    setup_0_m02_p_0.contact_law = make_slope_contact_law(slope=0)

    def inner_forces(x, t=None):
        return np.array([0, -0.2])

    setup_0_m02_p_0.inner_forces = inner_forces

    # setup2
    expected_displacement_vector_0_m02_p_0 = [
        [-v[0], -v[1]] for v in expected_displacement_vector_0_02_p_0
    ]

    test_suites.append((setup_0_m02_p_0, expected_displacement_vector_0_m02_p_0))

    # various changes

    @dataclass()
    class StaticSetup(StaticDisplacementProblem):
        mu_coef: ... = 4.58
        la_coef: ... = 3.33
        contact_law: ... = make_slope_contact_law(slope=2.71)

        @staticmethod
        def inner_forces(x, t=None):
            return np.array([0, -0.2])

        @staticmethod
        def outer_forces(x, t=None):
            return np.array([0.3, 0.0])

        boundaries: ... = BoundariesDescription(
            contact=lambda x: x[1] == 0, dirichlet=lambda x: x[0] == 0
        )

    mesh_descr = CrossMeshDescription(
        initial_position=None, max_element_perimeter=0.685, scale=[3.425, 1.37]
    )
    setup_var = StaticSetup(mesh_descr)
    # setup3
    expected_displacement_vector_var = [
        [0., 0.],
        [-0.02153888, 0.01375185],
        [-0.04850037, 0.05063871],
        [-0.07590935, 0.09732393],
        [-0.09874536, 0.15499642],
        [-0.12253538, 0.227189],
        [-0.19936814, 0.26117671],
        [-0.30550384, 0.2809148],
        [-0.2747237, 0.19398667],
        [-0.22878186, 0.13191358],
        [-0.17310884, 0.08301521],
        [-0.10282545, 0.04293346],
        [0., 0.],
        [0., 0.],
    ]

    test_suites.append((setup_var, expected_displacement_vector_var))

    return test_suites


@pytest.mark.parametrize("setup, expected_displacement_vector", generate_test_suits())
def test_static_solver(solving_method, opt_method, setup, expected_displacement_vector):
    runner = StaticSolver(setup, solving_method)
    result = runner.solve(initial_displacement=setup.initial_displacement, method=opt_method)

    displacement = result.body.mesh.nodes[:] - result.displaced_nodes[:]
    std_ids = standard_boundary_nodes(runner.body.mesh.nodes, runner.body.mesh.elements)

    # print result
    np.set_printoptions(precision=8, suppress=True)
    print(repr(displacement[std_ids]))

    np.testing.assert_array_almost_equal(
        displacement[std_ids], expected_displacement_vector, decimal=2  # TODO: increase precision
    )
