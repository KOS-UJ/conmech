import numpy as np
import pytest

from conmech.simulations.problem_solver import Static as StaticProblem
from examples.Jureczka_and_Ochal_2019 import StaticSetup
from tests.regression.std_boundary import standard_boundary_nodes


@pytest.fixture(params=[
    "global optimization",
    "schur"
])
def solving_method(request):
    return request.param


def test_global_optimization_solver(solving_method):
    expected_displacement_vector = \
        [[0., 0.],
         [0.04262992, 0.03218287],
         [0.06807502, 0.03495997],
         [0.08973783, 0.03809627],
         [0.10690461, 0.03891567],
         [0.1191638, 0.03825525],
         [0.12634279, 0.0356922],
         [0.12834505, 0.03062082],
         [0.12591354, 0.01632305],
         [0.14742755, 0.02758993],
         [0.16265202, 0.03806868],
         [0.17271473, 0.0452548],
         [0.17894004, 0.04771795],
         [0.17526471, 0.05199612],
         [0.16444484, 0.05331317],
         [0.14729425, 0.05081786],
         [0.12449315, 0.04438956],
         [0.09667483, 0.03395565],
         [0.06512335, 0.01972973],
         [0.03222816, 0.00343723],
         [0., 0.],
         [0., 0.],
         [0., 0.],
         [0., 0.]]
    setup = StaticSetup()
    runner = StaticProblem(setup, solving_method)
    result = runner.solve(fixed_point_abs_tol=0.001,
                          initial_displacement=setup.initial_displacement)

    displacement = result.mesh.initial_nodes[:] - result.displaced_points[:]
    std_ids = standard_boundary_nodes(runner.mesh.initial_nodes, runner.mesh.elements)

    # print result
    np.set_printoptions(precision=8, suppress=True)
    print(repr(displacement[std_ids]))

    np.testing.assert_array_almost_equal(
        displacement[std_ids], expected_displacement_vector, decimal=3)
