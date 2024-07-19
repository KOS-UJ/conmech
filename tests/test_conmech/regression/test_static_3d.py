"""
Created at 21.08.2019
"""

from dataclasses import dataclass

import numpy as np
import pytest

from conmech.mesh.boundaries_description import BoundariesDescription
from conmech.scenarios.problems import StaticDisplacementProblem
from conmech.simulations.problem_solver import StaticSolver
from conmech.properties.mesh_description import CrossMeshDescription, \
    CubeMeshDescription
from conmech.dynamics.contact.p_slope_contact_law import make_slope_contact_law
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
            return  0 * x

        boundaries: ... = BoundariesDescription(
            contact=lambda x: x[1] == 0, dirichlet=lambda x: x[0] == 0
        )

    mesh_descr = CubeMeshDescription(initial_position=None)
    setup_m02_m02 = StaticSetup(mesh_descr)

    expected_displacement_vector_m02_m02 = [
        [0., 0., 0.],
        [0., 0., 0.],
        [0., 0., 0.],
        [0.00886805, 0.00759973, 0.01309122],
        [0.01251517, 0.01304646, 0.02180133],
        [0.0047388, 0.01377955, 0.02134457],
        [-0.00179084, 0.01481824, 0.02136016],
        [0.00211174, 0.01387662, 0.02174084],
        [0.00165348, 0.00748375, 0.01153221],
        [-0.00196903, 0.00853421, 0.01211016],
        [0., 0., 0.],
        [0., 0., 0.],
        [-0.00548426, 0.00962111, 0.01202977],
        [-0.0060029, 0.01529378, 0.02094435],
        [0.00087541, 0.01483005, 0.02075665],
        [-0.00010425, 0.00850936, 0.01158434],
        [0., 0., 0.],
        [0., 0., 0.],
        [0.00577241, 0.00703007, 0.01156364],
        [0.00711386, 0.0132022, 0.02045272],
        [0., 0., 0.],
        [0., 0., 0.],
        [0.01268809, 0.0081164, 0.0137591],
        [0.01424769, 0.01004692, 0.02280081],
        [0.00928219, 0.01244232, 0.02200091],
        [0.00655036, 0.00806369, 0.01241398],
    ]

    test_suites.append((setup_m02_m02, expected_displacement_vector_m02_m02))

    # p = 0 and opposite forces

    setup_0_02_p_0 = StaticSetup(mesh_descr)
    setup_0_02_p_0.contact_law = make_slope_contact_law(slope=0)

    def inner_forces(x, t=None):
        result = 0.2 * x
        result[0] = 0.
        return result

    setup_0_02_p_0.inner_forces = inner_forces

    expected_displacement_vector_0_02_p_0 = [
        [0., 0., 0.],
        [0., 0., 0.],
        [0., 0., 0.],
        [-0.00521917, -0.01197369, -0.01232288],
        [-0.00671187, -0.02140411, -0.02124774],
        [0.00035836, -0.02143418, -0.02143418],
        [0.0075367, -0.02140411, -0.02218005],
        [0.00039165, -0.02055906, -0.02130339],
        [0.00016529, -0.01087314, -0.01172428],
        [0.00563059, -0.01197369, -0.01291526],
        [0., 0., 0.],
        [0., 0., 0.],
        [0.01132733, -0.01349373, -0.01349373],
        [0.01346888, -0.02202116, -0.02202116],
        [0.0075367, -0.02218005, -0.02140411],
        [0.00563059, -0.01291526, -0.01197369],
        [0., 0., 0.],
        [0., 0., 0.],
        [0.00016529, -0.01172428, -0.01087314],
        [0.00039165, -0.02130339, -0.02055906],
        [0., 0., 0.],
        [0., 0., 0.],
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
        result[0] = 0.
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
