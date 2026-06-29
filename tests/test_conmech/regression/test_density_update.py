"""
Created at 06.09.2023
"""

from dataclasses import dataclass

import numpy as np
import pytest

from conmech.mesh.boundaries_description import BoundariesDescription
from conmech.scenarios.problems import StaticDisplacementProblem
from conmech.simulations.problem_solver import NonHomogenousSolver
from conmech.properties.mesh_description import CrossMeshDescription
from conmech.dynamics.contact.relu_slope_contact_law import make_slope_contact_law
from tests.test_conmech.regression.std_boundary import standard_boundary_nodes


@pytest.fixture(params=["direct", "global optimization", "schur"])
def solving_method(request):
    return request.param


def get_elem_centers(runner: NonHomogenousSolver):
    elem_centers = np.empty(shape=(len(runner.body.mesh.elements), 2))
    for idx, elem in enumerate(runner.body.mesh.elements):
        verts = runner.body.mesh.nodes[elem]
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

        boundaries: ... = BoundariesDescription(
            contact=lambda x: x[1] == 0, dirichlet=lambda x: x[0] == 0
        )

    mesh_descr = CrossMeshDescription(
        initial_position=None, max_element_perimeter=0.25, scale=[2.5, 1]
    )
    setup_1 = StaticSetup(mesh_descr)

    expected_displacement_vectors_1 = [
        # setup0
        [
            [0.0, 0.0],
            [-0.02924609, -0.02413485],
            [-0.04921886, -0.0440495],
            [-0.06407169, -0.06514673],
            [-0.07537527, -0.08822226],
            [-0.11243257, -0.12160615],
            [-0.13607138, -0.14675812],
            [-0.15083027, -0.16731417],
            [-0.15844894, -0.18441889],
            [-0.15969834, -0.19900766],
            [-0.15479221, -0.21233831],
            [-0.14093197, -0.23364477],
            [-0.12409535, -0.24904007],
            [-0.10442407, -0.25827601],
            [-0.08250543, -0.26128892],
            [-0.07949436, -0.23763004],
            [-0.07031114, -0.21161755],
            [-0.05541532, -0.18227502],
            [-0.03620864, -0.14898295],
            [-0.01529128, -0.11231143],
            [0.00363429, -0.07955489],
            [0.00708805, -0.05653926],
            [0.00883508, -0.03388738],
            [0.007045, -0.01408388],
            [0.0, 0.0],
            [0.0, 0.0],
            [0.0, 0.0],
            [0.0, 0.0],
        ],
        # setup1
        [
            [0.0, 0.0],
            [-0.03000135, -0.02469796],
            [-0.05053746, -0.04534133],
            [-0.06569341, -0.06736692],
            [-0.07694982, -0.09069259],
            [-0.09217092, -0.11629417],
            [-0.10206127, -0.13955996],
            [-0.10812792, -0.1609411],
            [-0.11117586, -0.18070981],
            [-0.11164102, -0.19927595],
            [-0.10963907, -0.2173685],
            [-0.09125423, -0.22603304],
            [-0.07154465, -0.23229488],
            [-0.05063347, -0.23603765],
            [-0.02882209, -0.23725298],
            [-0.02760779, -0.21476039],
            [-0.02389187, -0.19132156],
            [-0.01785745, -0.16647971],
            [-0.01007889, -0.13991551],
            [-0.00158458, -0.11174983],
            [0.0061491, -0.08384695],
            [0.00925043, -0.05905615],
            [0.01036785, -0.03519688],
            [0.00785572, -0.01462178],
            [0.0, 0.0],
            [0.0, 0.0],
            [0.0, 0.0],
            [0.0, 0.0],
        ],
        # setup2
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
            return np.array([0, 0.1]) if x[1] < 0.2 and x[0] >= 2.0 else np.array([0, 0])

        boundaries: ... = BoundariesDescription(
            contact=lambda x: x[1] == 0 and x[0] <= 1.0, dirichlet=lambda x: x[0] == 0
        )

    mesh_descr = CrossMeshDescription(
        initial_position=None, max_element_perimeter=0.5, scale=[2.5, 1]
    )
    setup_2 = StaticSetup(mesh_descr)

    expected_displacement_vectors_2 = [
        # setup3
        [
            [0.0, 0.0],
            [0.04618958, 0.03954446],
            [0.08306192, 0.11665549],
            [0.20096414, 0.32413195],
            [0.26614292, 0.62621382],
            [0.29242491, 0.94577544],
            [0.00416508, 0.90460469],
            [-0.26927487, 0.89272992],
            [-0.25649393, 0.60371902],
            [-0.1983749, 0.32574654],
            [-0.08307566, 0.11683613],
            [-0.0462062, 0.03954483],
            [0.0, 0.0],
            [0.0, 0.0],
        ],
        # setup4
        [
            [0.0, 0.0],
            [0.04624452, 0.03968952],
            [0.08281797, 0.11579863],
            [0.13057317, 0.24686878],
            [0.15654855, 0.41813314],
            [0.16707307, 0.59601605],
            [0.00165445, 0.57957066],
            [-0.15783674, 0.57479782],
            [-0.1527122, 0.40913347],
            [-0.12955284, 0.2475101],
            [-0.08281856, 0.1159435],
            [-0.04625718, 0.03969205],
            [0.0, 0.0],
            [0.0, 0.0],
        ],
        # setup5
        [
            [0.0, 0.0],
            [0.09165037, 0.10057917],
            [0.16064146, 0.33569807],
            [0.2058385, 0.66787976],
            [0.22830296, 1.05861752],
            [0.2350899, 1.46257643],
            [-0.16208351, 1.45532423],
            [-0.52254658, 1.44108317],
            [-0.50920152, 1.06017022],
            [-0.45275785, 0.68336219],
            [-0.34627313, 0.3623118],
            [-0.19077607, 0.14100217],
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

    displacement = result.displaced_nodes[:] - result.body.mesh.nodes[:]
    std_ids = standard_boundary_nodes(runner.body.mesh.nodes, runner.body.mesh.elements)

    # print result
    np.set_printoptions(precision=8, suppress=True)
    print(repr(displacement[std_ids]))

    np.testing.assert_array_almost_equal(
        displacement[std_ids], expected_displacement_vector, decimal=2
    )
