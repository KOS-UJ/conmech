import numpy as np
import pytest
from conmech.simulations.problem_solver import StaticSolver as StaticProblem
from conmech.properties.mesh_description import CrossMeshDescription
from examples.Jureczka_and_Ochal_2019 import StaticSetup
from tests.test_conmech.regression.std_boundary import standard_boundary_nodes


@pytest.fixture(params=["global optimization", "schur"])
def solving_method(request):
    return request.param


def test(solving_method):
    expected_displacement_vector = [
        [0.0, 0.0],
        [0.06078453, 0.03250193],
        [0.10404238, 0.03644206],
        [0.13873971, 0.04036119],
        [0.16570003, 0.04259187],
        [0.18495207, 0.04404475],
        [0.19660148, 0.0446417],
        [0.20075845, 0.0442943],
        [0.19765212, 0.04232086],
        [0.19967536, 0.05988723],
        [0.19991491, 0.07250985],
        [0.19784972, 0.08020974],
        [0.19323891, 0.08276982],
        [0.1897699, 0.07613559],
        [0.17937263, 0.06748543],
        [0.16222708, 0.05688091],
        [0.13856141, 0.04428154],
        [0.10898344, 0.0296097],
        [0.07489846, 0.01345543],
        [0.03838516, -0.00183673],
        [0.0, 0.0],
        [0.0, 0.0],
        [0.0, 0.0],
        [0.0, 0.0],
    ]

    mesh_descr = CrossMeshDescription(
        initial_position=None, max_element_perimeter=0.25, scale=[2, 1]
    )
    setup = StaticSetup(mesh_descr)
    runner = StaticProblem(setup, solving_method)
    result = runner.solve(
        initial_displacement=setup.initial_displacement,
        method="Powell" if solving_method == "schur" else "BFGS",
    )

    displacement = result.body.mesh.nodes[:] - result.displaced_nodes[:]
    std_ids = standard_boundary_nodes(runner.body.mesh.nodes, runner.body.mesh.elements)

    # print result
    np.set_printoptions(precision=8, suppress=True)
    print(repr(displacement[std_ids]))

    precision = 1 if solving_method != "global optimization" else 3
    np.testing.assert_array_almost_equal(
        displacement[std_ids], expected_displacement_vector, decimal=precision
    )
