import numpy as np
import pytest
from conmech.simulations.problem_solver import StaticSolver as StaticProblem
from conmech.properties.mesh_description import CrossMeshDescription
from examples.Jureczka_Ochal_2019 import StaticSetup
from tests.test_conmech.regression.std_boundary import standard_boundary_nodes


@pytest.fixture(params=["global optimization", "schur"])
def solving_method(request):
    return request.param


def test(solving_method):
    expected_displacement_vector = [
        [0.0, 0.0],
        [0.06050514, 0.02870868],
        [0.10409879, 0.03581729],
        [0.13876208, 0.03985382],
        [0.16570398, 0.04228241],
        [0.18497001, 0.04380692],
        [0.19665465, 0.04447322],
        [0.20087999, 0.04417727],
        [0.19785163, 0.04272112],
        [0.19956719, 0.06010641],
        [0.19966797, 0.07267026],
        [0.19746359, 0.08035769],
        [0.19270201, 0.08291694],
        [0.1892335, 0.07612806],
        [0.17884117, 0.06731372],
        [0.16171329, 0.05653334],
        [0.13810013, 0.04374914],
        [0.10864678, 0.02892239],
        [0.07476991, 0.01275361],
        [0.03844468, -0.00228974],
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
