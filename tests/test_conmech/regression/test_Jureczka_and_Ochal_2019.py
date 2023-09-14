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
        [0.03425259, 0.02042142],
        [0.05996324, 0.02664742],
        [0.08242059, 0.03067725],
        [0.10278965, 0.03311017],
        [0.12126769, 0.03481482],
        [0.13788175, 0.03610288],
        [0.15261347, 0.03712887],
        [0.1654474, 0.03795604],
        [0.17637907, 0.03860709],
        [0.18541298, 0.03908582],
        [0.19255815, 0.03938675],
        [0.19782477, 0.03949766],
        [0.20122288, 0.03939805],
        [0.20276421, 0.03905306],
        [0.20247181, 0.0384095],
        [0.20040875, 0.037322],
        [0.20149744, 0.04556468],
        [0.20222959, 0.0526955],
        [0.20253194, 0.05876645],
        [0.20232406, 0.06378807],
        [0.20154042, 0.06774787],
        [0.20013429, 0.07061872],
        [0.19808398, 0.07236208],
        [0.19540129, 0.0729355],
        [0.19452563, 0.06978214],
        [0.19189549, 0.06617879],
        [0.18754677, 0.06212332],
        [0.18150793, 0.05762658],
        [0.17379886, 0.05270484],
        [0.1644381, 0.04736932],
        [0.15345191, 0.04162382],
        [0.14088499, 0.0354679],
        [0.12681317, 0.02890669],
        [0.11135776, 0.02197101],
        [0.09470009, 0.01475316],
        [0.07709233, 0.00747065],
        [0.05885017, 0.000569],
        [0.04030411, -0.00500039],
        [0.02135662, -0.00767498],
        [0.0, 0.0],
        [0.0, 0.0],
        [0.0, 0.0],
        [0.0, 0.0],
        [0.0, 0.0],
        [0.0, 0.0],
        [0.0, 0.0],
        [0.0, 0.0],
    ]

    mesh_descr = CrossMeshDescription(
        initial_position=None, max_element_perimeter=0.125, scale=[2, 1]
    )
    setup = StaticSetup(mesh_descr)
    runner = StaticProblem(setup, solving_method)
    result = runner.solve(
        fixed_point_abs_tol=0.001, initial_displacement=setup.initial_displacement
    )

    displacement = result.body.mesh.nodes[:] - result.displaced_nodes[:]
    std_ids = standard_boundary_nodes(runner.body.mesh.nodes, runner.body.mesh.elements)

    # print result
    np.set_printoptions(precision=8, suppress=True)
    print(repr(displacement[std_ids]))

    precision = 2 if solving_method == "global optimization" else 3
    np.testing.assert_array_almost_equal(
        displacement[std_ids], expected_displacement_vector, decimal=precision
    )
