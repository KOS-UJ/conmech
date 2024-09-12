"""
Created at 21.08.2019
"""

from dataclasses import dataclass

import numpy as np
import pytest

from conmech.mesh.boundaries_description import BoundariesDescription
from conmech.scenarios.problems import StaticDisplacementProblem
from conmech.simulations.problem_solver import StaticSolver
from conmech.properties.mesh_description import (
    CrossMeshDescription,
    CubeMeshDescription,
)
from conmech.dynamics.contact.relu_slope_contact_law import make_slope_contact_law
from tests.test_conmech.regression.std_boundary import standard_boundary_nodes


@pytest.fixture(params=["global optimization"])
def solving_method(request):
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
            return -0.2 * x

        @staticmethod
        def outer_forces(x, t=None):
            return 0 * x

        boundaries: ... = BoundariesDescription(
            contact=lambda x: x[1] == 0, dirichlet=lambda x: x[0] == 0
        )

    mesh_descr = CubeMeshDescription(initial_position=None)
    setup_m02_m02 = StaticSetup(mesh_descr)

    expected_displacement_vector_m02_m02 = [
        [0.0, 0.0, 0.0],
        [0.0, 0.0, 0.0],
        [0.0, 0.0, 0.0],
        [0.00887781, 0.00938318, 0.01303338],
        [0.01254745, 0.0164677, 0.02173087],
        [0.00484257, 0.01694291, 0.02137632],
        [-0.00172692, 0.01755931, 0.02152949],
        [0.00322087, 0.01678243, 0.0214544],
        [0.00253174, 0.00921583, 0.01132137],
        [-0.00196061, 0.00997206, 0.01218131],
        [0.0, 0.0, 0.0],
        [0.0, 0.0, 0.0],
        [-0.0062681, 0.01092748, 0.0123151],
        [-0.00694282, 0.01796747, 0.021246],
        [-0.00016377, 0.01769789, 0.02100172],
        [-0.00086678, 0.0100699, 0.0117306],
        [0.0, 0.0, 0.0],
        [0.0, 0.0, 0.0],
        [0.00494897, 0.00864805, 0.0115685],
        [0.00615472, 0.0164003, 0.02059693],
        [0.0, 0.0, 0.0],
        [0.0, 0.0, 0.0],
        [0.01361542, 0.01025, 0.01362368],
        [0.01573552, 0.01474276, 0.02232607],
        [0.01065649, 0.01625437, 0.02177309],
        [0.00752115, 0.01009524, 0.0122507],
    ]

    test_suites.append((setup_m02_m02, expected_displacement_vector_m02_m02))

    # p = 0 and opposite forces

    setup_0_02_p_0 = StaticSetup(mesh_descr)
    setup_0_02_p_0.contact_law = make_slope_contact_law(slope=0)

    def inner_forces(x, t=None):
        result = 0.2 * x
        result[0] = 0.0
        return result

    setup_0_02_p_0.inner_forces = inner_forces

    expected_displacement_vector_0_02_p_0 = [
        [0.0, 0.0, 0.0],
        [0.0, 0.0, 0.0],
        [0.0, 0.0, 0.0],
        [-0.00521917, -0.01197369, -0.01232288],
        [-0.00671187, -0.02140411, -0.02124774],
        [0.00035836, -0.02143418, -0.02143418],
        [0.0075367, -0.02140411, -0.02218005],
        [0.00039165, -0.02055906, -0.02130339],
        [0.00016529, -0.01087314, -0.01172428],
        [0.00563059, -0.01197369, -0.01291526],
        [0.0, 0.0, 0.0],
        [0.0, 0.0, 0.0],
        [0.01132733, -0.01349373, -0.01349373],
        [0.01346888, -0.02202116, -0.02202116],
        [0.0075367, -0.02218005, -0.02140411],
        [0.00563059, -0.01291526, -0.01197369],
        [0.0, 0.0, 0.0],
        [0.0, 0.0, 0.0],
        [0.00016529, -0.01172428, -0.01087314],
        [0.00039165, -0.02130339, -0.02055906],
        [0.0, 0.0, 0.0],
        [0.0, 0.0, 0.0],
        [-0.01099676, -0.01264259, -0.01264259],
        [-0.01268559, -0.02127684, -0.02127684],
        [-0.00671187, -0.02124774, -0.02140411],
        [-0.00521917, -0.01232288, -0.01197369],
    ]

    test_suites.append((setup_0_02_p_0, expected_displacement_vector_0_02_p_0))

    # p = 0

    setup_0_m02_p_0 = StaticSetup(mesh_descr)
    setup_0_m02_p_0.contact_law = make_slope_contact_law(slope=0)

    def inner_forces(x, t=None):
        result = -0.2 * x
        result[0] = 0.0
        return result

    setup_0_m02_p_0.inner_forces = inner_forces

    expected_displacement_vector_0_m02_p_0 = [
        [-v[0], -v[1], -v[2]] for v in expected_displacement_vector_0_02_p_0
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

    return test_suites


@pytest.mark.parametrize("setup, expected_displacement_vector", generate_test_suits())
def test_static_solver(solving_method, setup, expected_displacement_vector):
    runner = StaticSolver(setup, solving_method)
    result = runner.solve(initial_displacement=setup.initial_displacement)

    displacement = result.body.mesh.nodes[:] - result.displaced_nodes[:]
    std_ids = standard_boundary_nodes(runner.body.mesh.nodes, runner.body.mesh.elements)

    # print result
    np.set_printoptions(precision=8, suppress=True)
    print(repr(displacement[std_ids]))

    np.testing.assert_array_almost_equal(
        displacement[std_ids], expected_displacement_vector, decimal=3
    )
