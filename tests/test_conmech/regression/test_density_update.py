"""
Created at 06.09.2023
"""
from dataclasses import dataclass

import numpy as np
import pytest

from conmech.mesh.boundaries_description import BoundariesDescription
from conmech.scenarios.problems import StaticDisplacementProblem
from conmech.simulations.problem_solver import NonHomogenousSolver
from examples.p_slope_contact_law import make_slope_contact_law
from tests.test_conmech.regression.std_boundary import standard_boundary_nodes


@pytest.fixture(params=["direct", "global optimization", "schur"])
def solving_method(request):
    return request.param


def get_elem_centers(runner: NonHomogenousSolver):
    elem_centers = np.empty(shape=(len(runner.body.mesh.elements), 2))
    for idx, elem in enumerate(runner.body.mesh.elements):
        verts = runner.body.mesh.initial_nodes[elem]
        elem_centers[idx] = np.sum(verts, axis=0) / len(elem)
    return elem_centers


def generate_test_suits():
    test_suites = []

    density_functions = [
        lambda x: 1 if x[0] < 1 else 0.2,
        lambda x: 1 if x[0] < 1 else 0.5,
        lambda x: 1 if x[1] < 0.5 else 0.1
        ]

    @dataclass()
    class StaticSetup(StaticDisplacementProblem):
        grid_height: ... = 1
        elements_number: ... = (2, 5)
        mu_coef: ... = 4
        la_coef: ... = 4
        contact_law: ... = make_slope_contact_law(slope=1)

        @staticmethod
        def inner_forces(x, t=None):
            return np.array([-0.2, -0.2])

        @staticmethod
        def outer_forces(x, t=None):
            return np.array([0, 0])

        @staticmethod
        def friction_bound(u_nu):
            return 0

        boundaries: ... = BoundariesDescription(
            contact=lambda x: x[1] == 0, dirichlet=lambda x: x[0] == 0
        )

    setup_1 = StaticSetup(mesh_type="cross")

    expected_displacement_vectors_1 = [
        
        [[ 0.        ,  0.        ],
        [-0.04635128, -0.04232486],
        [-0.0730364 , -0.08338891],
        [-0.13669534, -0.14426146],
        [-0.16149804, -0.1831861 ],
        [-0.15864615, -0.21527157],
        [-0.12462532, -0.2509446 ],
        [-0.0800199 , -0.2628315 ],
        [-0.06831759, -0.20998748],
        [-0.0365764 , -0.14430532],
        [ 0.00182313, -0.07428595],
        [ 0.00618507, -0.03142759],
        [ 0.        ,  0.        ],
        [ 0.        ,  0.        ]],
        
        [[ 0.        ,  0.        ],
        [-0.04777782, -0.04371089],
        [-0.07533447, -0.08647969],
        [-0.10241762, -0.13597733],
        [-0.11294151, -0.17846116],
        [-0.11188724, -0.21782305],
        [-0.07167752, -0.23206402],
        [-0.02710726, -0.23680208],
        [-0.02245196, -0.18888777],
        [-0.00991831, -0.1357546 ],
        [ 0.00482366, -0.07919934],
        [ 0.0078284 , -0.03277353],
        [ 0.        ,  0.        ],
        [ 0.        ,  0.        ]],

       [[ 0.        ,  0.        ],
        [-0.05879403, -0.06160514],
        [-0.08474773, -0.11756289],
        [-0.09332965, -0.15399231],
        [-0.09412659, -0.16768477],
        [-0.09241176, -0.1679926 ],
        [-0.09383849, -0.17246538],
        [-0.13116408, -0.18193134],
        [-0.12027193, -0.1863892 ],
        [-0.09302902, -0.16389214],
        [-0.05856332, -0.12162893],
        [-0.02929081, -0.06844951],
        [ 0.        ,  0.        ],
        [ 0.        ,  0.        ]]
    ]

    for fun, expected in zip(density_functions, expected_displacement_vectors_1):
        test_suites.append((setup_1, fun, expected))


    @dataclass()
    class StaticSetup(StaticDisplacementProblem):
        grid_height: ... = 1
        elements_number: ... = (2, 5)
        mu_coef: ... = 4
        la_coef: ... = 4
        contact_law: ... = make_slope_contact_law(slope=1)

        @staticmethod
        def inner_forces(x, t=None):
            return np.array([0, 0])

        @staticmethod
        def outer_forces(x, t=None):
            return np.array([0, 0.1]) if x[1] < 0.2 and x[0] > 2.2 else np.array([0, 0])

        @staticmethod
        def friction_bound(u_nu: float) -> float:
            return 0

        boundaries: ... = BoundariesDescription(
            contact=lambda x: x[1] == 0 and x[0] < 0.2, dirichlet=lambda x: x[0] == 0
        )


    setup_2 = StaticSetup(mesh_type="cross")

    expected_displacement_vectors_2 = [
        
        [[ 0.        ,  0.        ],
        [ 0.02599293,  0.021492  ],
        [ 0.04747832,  0.06462195],
        [ 0.12073424,  0.18288368],
        [ 0.16852813,  0.35124948],
        [ 0.19426259,  0.5945299 ],
        [-0.00279276,  0.54908258],
        [-0.17303802,  0.53807316],
        [-0.16239225,  0.35466071],
        [-0.12059053,  0.18422091],
        [-0.04751902,  0.06465119],
        [-0.02599838,  0.02148409],
        [ 0.        ,  0.        ],
        [ 0.        ,  0.        ]],
        
        [[ 0.        ,  0.        ],
        [ 0.02602551,  0.0215782 ],
        [ 0.04733487,  0.06410662],
        [ 0.07700301,  0.1388405 ],
        [ 0.09606589,  0.2350403 ],
        [ 0.10636675,  0.36098208],
        [-0.00111825,  0.34281676],
        [-0.09787904,  0.33839926],
        [-0.09361354,  0.23640489],
        [-0.07694793,  0.13937786],
        [-0.04736453,  0.06412507],
        [-0.02602924,  0.02157238],
        [ 0.        ,  0.        ],
        [ 0.        ,  0.        ]],
        
        [[ 0.        ,  0.        ],
        [ 0.05135096,  0.05502034],
        [ 0.09153136,  0.18665172],
        [ 0.11991624,  0.37669138],
        [ 0.13705379,  0.60465541],
        [ 0.14408795,  0.85793818],
        [-0.10038042,  0.85018255],
        [-0.31362968,  0.83921941],
        [-0.30391213,  0.61065957],
        [-0.26572229,  0.38741854],
        [-0.19960097,  0.20262106],
        [-0.1086065 ,  0.07793528],
        [ 0.        ,  0.        ],
        [ 0.        ,  0.        ]]
    ]

    for fun, expected in zip(density_functions, expected_displacement_vectors_2):
        test_suites.append((setup_2, fun, expected))


    return test_suites


# TODO find out why "schur" method gives different result in first test scenario
@pytest.mark.parametrize("setup, density_func, expected_displacement_vector", generate_test_suits())
def test_direct_solver(solving_method, setup, density_func, expected_displacement_vector):
    runner = NonHomogenousSolver(setup, solving_method)
    elem_centers = get_elem_centers(runner)
    elements_density = np.asarray([density_func(x) for x in elem_centers])
    runner.update_density(elements_density)

    result = runner.solve(initial_displacement=setup.initial_displacement)

    displacement = result.displaced_nodes[:] - result.body.mesh.initial_nodes[:]
    std_ids = standard_boundary_nodes(runner.body.mesh.initial_nodes, runner.body.mesh.elements)

    # print result
    np.set_printoptions(precision=8, suppress=True)
    print(repr(displacement[std_ids]))

    np.testing.assert_array_almost_equal(
        displacement[std_ids], expected_displacement_vector, decimal=3
    )
