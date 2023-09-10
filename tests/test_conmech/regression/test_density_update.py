"""
Created at 06.09.2023
"""
from dataclasses import dataclass

import numpy as np
import pytest

from conmech.mesh.boundaries_description import BoundariesDescription
from conmech.scenarios.problems import StaticDisplacementProblem
from conmech.simulations.problem_solver import NonHomogenousSolver
from conmech.properties.mesh_properties import CrossMeshDescription
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
        lambda x: 1 if x[1] < 0.5 else 0.1,
    ]

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

        @staticmethod
        def friction_bound(u_nu):
            return 0

        boundaries: ... = BoundariesDescription(
            contact=lambda x: x[1] == 0, dirichlet=lambda x: x[0] == 0
        )

    mesh_descr = CrossMeshDescription(
        initial_position=None, max_element_perimeter=0.25, scale=[2.5, 1]
    )
    setup_1 = StaticSetup(mesh_descr)

    expected_displacement_vectors_1 = [
        [
            [0.0, 0.0],
            [-0.02924737, -0.02414377],
            [-0.04921958, -0.04405341],
            [-0.06407238, -0.06515026],
            [-0.07537587, -0.08822544],
            [-0.11243269, -0.12160857],
            [-0.13607121, -0.14676002],
            [-0.15082993, -0.16731555],
            [-0.15844851, -0.18441979],
            [-0.15969787, -0.19900811],
            [-0.15479172, -0.21233831],
            [-0.14093195, -0.23364479],
            [-0.12409582, -0.2490401],
            [-0.10442504, -0.25827603],
            [-0.08250694, -0.26128894],
            [-0.07949587, -0.2376306],
            [-0.07031265, -0.21161868],
            [-0.05541679, -0.18227676],
            [-0.03620997, -0.14898536],
            [-0.01529227, -0.11231454],
            [0.00363405, -0.07955863],
            [0.00708811, -0.05654278],
            [0.00883554, -0.03389028],
            [0.00704566, -0.01408556],
            [0.0, 0.0],
            [0.0, 0.0],
            [0.0, 0.0],
            [0.0, 0.0],
        ],
        [
            [0.0, 0.0],
            [-0.03000257, -0.02470704],
            [-0.05053805, -0.04534519],
            [-0.06569392, -0.06737029],
            [-0.07695024, -0.09069556],
            [-0.09217111, -0.11629674],
            [-0.10206127, -0.13956213],
            [-0.1081278, -0.16094285],
            [-0.11117567, -0.18071113],
            [-0.11164081, -0.19927686],
            [-0.10963884, -0.217369],
            [-0.0912544, -0.22603357],
            [-0.07154523, -0.23229543],
            [-0.05063447, -0.23603821],
            [-0.02882352, -0.23725354],
            [-0.02760922, -0.21476139],
            [-0.0238933, -0.191323],
            [-0.01785883, -0.16648163],
            [-0.01008018, -0.13991794],
            [-0.00158566, -0.11175275],
            [0.00614847, -0.08385029],
            [0.0092502, -0.05905948],
            [0.01036813, -0.03519972],
            [0.00785629, -0.01462345],
            [0.0, 0.0],
            [0.0, 0.0],
            [0.0, 0.0],
            [0.0, 0.0],
        ],
        [
            [0.0, 0.0],
            [-0.03822041, -0.03387305],
            [-0.06244525, -0.06598509],
            [-0.07713491, -0.09797919],
            [-0.08548996, -0.1251864],
            [-0.08968083, -0.14580808],
            [-0.09118055, -0.15937592],
            [-0.09098958, -0.16626868],
            [-0.08988218, -0.16740786],
            [-0.08857686, -0.16432274],
            [-0.08739773, -0.15949831],
            [-0.09191057, -0.16218342],
            [-0.09804682, -0.16291048],
            [-0.12615713, -0.16800452],
            [-0.14553212, -0.17224207],
            [-0.14095383, -0.1839543],
            [-0.12994638, -0.18549517],
            [-0.11539392, -0.18015809],
            [-0.09765657, -0.16919595],
            [-0.07774314, -0.15223532],
            [-0.0572547, -0.12929652],
            [-0.03789009, -0.10136915],
            [-0.02063662, -0.07026397],
            [-0.00555871, -0.03545273],
            [0.0, 0.0],
            [0.0, 0.0],
            [0.0, 0.0],
            [0.0, 0.0],
        ],
    ]

    for fun, expected in zip(density_functions, expected_displacement_vectors_1):
        test_suites.append((setup_1, fun, expected))

    @dataclass()
    class StaticSetup(StaticDisplacementProblem):
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

    mesh_descr = CrossMeshDescription(
        initial_position=None, max_element_perimeter=0.5, scale=[2.5, 1]
    )
    setup_2 = StaticSetup(mesh_descr)

    expected_displacement_vectors_2 = [
        [
            [0.0, 0.0],
            [0.02599293, 0.021492],
            [0.04747832, 0.06462195],
            [0.12073424, 0.18288368],
            [0.16852813, 0.35124948],
            [0.19426259, 0.5945299],
            [-0.00279276, 0.54908258],
            [-0.17303802, 0.53807316],
            [-0.16239225, 0.35466071],
            [-0.12059053, 0.18422091],
            [-0.04751902, 0.06465119],
            [-0.02599838, 0.02148409],
            [0.0, 0.0],
            [0.0, 0.0],
        ],
        [
            [0.0, 0.0],
            [0.02602551, 0.0215782],
            [0.04733487, 0.06410662],
            [0.07700301, 0.1388405],
            [0.09606589, 0.2350403],
            [0.10636675, 0.36098208],
            [-0.00111825, 0.34281676],
            [-0.09787904, 0.33839926],
            [-0.09361354, 0.23640489],
            [-0.07694793, 0.13937786],
            [-0.04736453, 0.06412507],
            [-0.02602924, 0.02157238],
            [0.0, 0.0],
            [0.0, 0.0],
        ],
        [
            [0.0, 0.0],
            [0.05135096, 0.05502034],
            [0.09153136, 0.18665172],
            [0.11991624, 0.37669138],
            [0.13705379, 0.60465541],
            [0.14408795, 0.85793818],
            [-0.10038042, 0.85018255],
            [-0.31362968, 0.83921941],
            [-0.30391213, 0.61065957],
            [-0.26572229, 0.38741854],
            [-0.19960097, 0.20262106],
            [-0.1086065, 0.07793528],
            [0.0, 0.0],
            [0.0, 0.0],
        ],
    ]

    for fun, expected in zip(density_functions, expected_displacement_vectors_2):
        test_suites.append((setup_2, fun, expected))

    return test_suites


@pytest.mark.parametrize("setup, density_func, expected_displacement_vector", generate_test_suits())
def test_nonhomogenous_solver(solving_method, setup, density_func, expected_displacement_vector):
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
