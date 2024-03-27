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
        [0., 0.],
        [0.03522857, 0.02154723],
        [0.06134879, 0.02852184],
        [0.08393532, 0.03320384],
        [0.10426168, 0.03618088],
        [0.12257757, 0.03834405],
        [0.13894498, 0.04001477],
        [0.15337147, 0.04135569],
        [0.1658617, 0.04243747],
        [0.17642638, 0.04329012],
        [0.18508084, 0.04392449],
        [0.19184122, 0.04434147],
        [0.1967216, 0.04453428],
        [0.19973326, 0.04448642],
        [0.20088703, 0.0441648],
        [0.20020383, 0.04351448],
        [0.19774689, 0.04237685],
        [0.19891282, 0.05173695],
        [0.19972586, 0.05984292],
        [0.20010308, 0.06674484],
        [0.19996122, 0.07244883],
        [0.19923657, 0.07693874],
        [0.19788669, 0.08018577],
        [0.19589475, 0.08215244],
        [0.19327284, 0.08279996],
        [0.19239862, 0.07968588],
        [0.18976624, 0.07607903],
        [0.18540719, 0.07197493],
        [0.17934638, 0.06737894],
        [0.1716013, 0.0622999],
        [0.16218933, 0.05673946],
        [0.15113744, 0.05068978],
        [0.13849374, 0.04413655],
        [0.12434129, 0.03706901],
        [0.10881389, 0.02950038],
        [0.09211286, 0.02150421],
        [0.0745211, 0.01327832],
        [0.05640314, 0.00524877],
        [0.03817266, -0.00165521],
        [0.01990921, -0.00585183],
        [0., 0.],
        [0., 0.],
        [0., 0.],
        [0., 0.],
        [0., 0.],
        [0., 0.],
        [0., 0.],
        [0., 0.]
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
