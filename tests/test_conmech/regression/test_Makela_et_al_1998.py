# CONMECH @ Jagiellonian University in Kraków
#
# Copyright (C) 2024  Piotr Bartman-Szwarc <piotr.bartman@uj.edu.pl>
#
# This program is free software; you can redistribute it and/or
# modify it under the terms of the GNU General Public License
# as published by the Free Software Foundation; either version 3
# of the License, or (at your option) any later version.
#
# This program is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
# GNU General Public License for more details.
#
# You should have received a copy of the GNU General Public License
# along with this program; if not, write to the Free Software
# Foundation, Inc., 51 Franklin Street, Fifth Floor, Boston, MA  02110-1301,
# USA.
from dataclasses import dataclass

import numpy as np
import pytest

from conmech.mesh.boundaries_description import BoundariesDescription
from conmech.scenarios.problems import StaticDisplacementProblem
from conmech.simulations.problem_solver import StaticSolver
from conmech.properties.mesh_description import CrossMeshDescription, RectangleMeshDescription
from conmech.solvers.optimization.optimization import Optimization
from examples.Makela_et_al_1998 import MMLV99
from tests.test_conmech.regression.std_boundary import standard_boundary_nodes


@pytest.fixture(params=["schur"])
def solving_method(request):
    return request.param


def generate_test_suits():
    test_suites = []

    optimization_mtd_pow = "Powell"

    expected_displacement_vector_pow = [
        [0., 0.],
        [-0.00001052, -0.00000833],
        [-0.00001962, -0.00001745],
        [-0.00002745, -0.00003174],
        [-0.0000345, -0.0000481],
        [-0.00004035, -0.00006865],
        [-0.00004552, -0.00009066],
        [-0.00004975, -0.0001157],
        [-0.00005346, -0.00014177],
        [-0.00005645, -0.00017],
        [-0.00005907, -0.00019895],
        [-0.00006118, -0.00022943],
        [-0.00006303, -0.00026042],
        [-0.00006451, -0.00029252],
        [-0.00006582, -0.000325],
        [-0.00006687, -0.00035831],
        [-0.00006781, -0.00039192],
        [-0.00006855, -0.00042619],
        [-0.00006919, -0.0004607],
        [-0.00006967, -0.00049582],
        [-0.00007003, -0.00053111],
        [-0.00007025, -0.00056658],
        [-0.00007041, -0.0006021],
        [-0.00007047, -0.00063776],
        [-0.00007051, -0.00067346],
        [-0.00007049, -0.00070925],
        [-0.00007046, -0.00074507],
        [-0.00007041, -0.00078096],
        [-0.00007035, -0.00081689],
        [-0.00007028, -0.0008529],
        [-0.00007019, -0.00088894],
        [-0.00007006, -0.00092505],
        [-0.00006991, -0.00096119],
        [-0.00006972, -0.00099732],
        [-0.00003993, -0.00099771],
        [-0.00001013, -0.00099814],
        [0.00001963, -0.00099859],
        [0.00004954, -0.00099912],
        [0.00007947, -0.0009997],
        [0.00007922, -0.00096314],
        [0.00007891, -0.00092685],
        [0.00007862, -0.00089072],
        [0.00007832, -0.00085463],
        [0.00007802, -0.00081863],
        [0.00007773, -0.00078266],
        [0.00007742, -0.00074677],
        [0.00007711, -0.0007109],
        [0.00007677, -0.00067512],
        [0.00007641, -0.00063936],
        [0.00007599, -0.00060371],
        [0.00007552, -0.00056812],
        [0.00007493, -0.00053273],
        [0.00007425, -0.00049744],
        [0.00007341, -0.00046252],
        [0.00007246, -0.00042781],
        [0.00007135, -0.00039366],
        [0.00007013, -0.00035978],
        [0.00006871, -0.00032663],
        [0.00006715, -0.00029382],
        [0.00006531, -0.00026197],
        [0.00006325, -0.00023055],
        [0.00006079, -0.00020043],
        [0.00005801, -0.00017093],
        [0.00005465, -0.00014324],
        [0.00005085, -0.00011642],
        [0.00004624, -0.00009215],
        [0.00004105, -0.00006911],
        [0.00003479, -0.00004963],
        [0.00002775, -0.00003184],
        [0.00001932, -0.00001891],
        [0.00000961, -0.0000077],
        [0., 0.],
        [0., 0.],
        [0., 0.],
        [0., 0.],
        [0., 0.],
    ]
    test_suites.append((optimization_mtd_pow, expected_displacement_vector_pow))

    optimization_mtd_subg = "subgradient"
    expected_displacement_vector_subg = [
       [ 0.        ,  0.        ],
       [-0.00006594, -0.00004315],
       [-0.00012749, -0.00009338],
       [-0.00018649, -0.00018164],
       [-0.00024325, -0.0002873 ],
       [-0.00029591, -0.00043023],
       [-0.00034592, -0.00058795],
       [-0.00039183, -0.00077834],
       [-0.00043521, -0.00098149],
       [-0.00047474, -0.00121284],
       [-0.00051191, -0.00145513],
       [-0.0005455 , -0.00172143],
       [-0.0005769 , -0.00199696],
       [-0.00060501, -0.00229263],
       [-0.00063112, -0.00259596],
       [-0.00065425, -0.00291588],
       [-0.00067557, -0.00324204],
       [-0.00069422, -0.00358162],
       [-0.00071126, -0.00392617],
       [-0.00072592, -0.00428129],
       [-0.00073916, -0.00464026],
       [-0.00075031, -0.0050073 ],
       [-0.00076022, -0.0053772 ],
       [-0.00076831, -0.00575298],
       [-0.00077534, -0.00613077],
       [-0.00078084, -0.00651256],
       [-0.00078546, -0.00689564],
       [-0.00078884, -0.00728117],
       [-0.00079152, -0.00766739],
       [-0.00079327, -0.00805485],
       [-0.00079451, -0.00844256],
       [-0.00079513, -0.00883062],
       [-0.00079547, -0.00921861],
       [-0.0007956 , -0.00960656],
       [-0.00047553, -0.00960645],
       [-0.00015576, -0.00960649],
       [ 0.0001639 , -0.00960666],
       [ 0.00048371, -0.00960712],
       [ 0.0008037 , -0.00960768],
       [ 0.00080342, -0.0092195 ],
       [ 0.00080296, -0.00883138],
       [ 0.0008021 , -0.00844342],
       [ 0.00080088, -0.00805553],
       [ 0.00079888, -0.0076684 ],
       [ 0.0007963 , -0.00728171],
       [ 0.00079264, -0.00689683],
       [ 0.00078819, -0.00651292],
       [ 0.00078235, -0.0061322 ],
       [ 0.00077554, -0.00575311],
       [ 0.00076705, -0.00537893],
       [ 0.00075739, -0.00500716],
       [ 0.00074577, -0.00464232],
       [ 0.00073279, -0.00428083],
       [ 0.00071756, -0.00392862],
       [ 0.00070079, -0.00358081],
       [ 0.00068145, -0.00324495],
       [ 0.00066038, -0.00291468],
       [ 0.00063644, -0.00259938],
       [ 0.00061058, -0.00229101],
       [ 0.00058154, -0.00200098],
       [ 0.00055038, -0.00171937],
       [ 0.00051574, -0.0014598 ],
       [ 0.00047881, -0.0012103 ],
       [ 0.00043813, -0.00098686],
       [ 0.00039498, -0.00077525],
       [ 0.00034784, -0.00059401],
       [ 0.00029808, -0.00042646],
       [ 0.00024407, -0.00029379],
       [ 0.00018721, -0.00017669],
       [ 0.00012562, -0.00009923],
       [ 0.00005962, -0.00003655],
       [ 0.        ,  0.        ],
       [ 0.        ,  0.        ],
       [ 0.        ,  0.        ],
       [ 0.        ,  0.        ],
       [ 0.        ,  0.        ],
    ]

    test_suites.append((optimization_mtd_subg, expected_displacement_vector_subg))

    return test_suites


@pytest.mark.parametrize("optimization_method, expected_displacement_vector", generate_test_suits())
def test_static_solver(solving_method, optimization_method, expected_displacement_vector):
    mesh_density = 4
    kN = 1000
    mm = 0.001
    E = 1.378e8 * kN
    kappa = 0.3
    surface = 5 * mm * 80 * mm

    @dataclass()
    class StaticSetup(StaticDisplacementProblem):
        grid_height: ... = 10 * mm
        elements_number: ... = (mesh_density, 8 * mesh_density)
        mu_coef: ... = (E * surface) / (2 * (1 + kappa))
        la_coef: ... = ((E * surface) * kappa) / ((1 + kappa) * (1 - 2 * kappa))
        contact_law: ... = MMLV99()

        @staticmethod
        def inner_forces(x, t=None):
            return np.array([0.0, 0.0])

        @staticmethod
        def outer_forces(x, t=None):
            if x[1] >= 0.0099:
                return np.array([0, 26.2e3 * kN * surface])
            return np.array([0, 0])

        @staticmethod
        def friction_bound(u_nu: float) -> float:
            return 0.0

        boundaries: ... = BoundariesDescription(
            contact=lambda x: x[1] == 0, dirichlet=lambda x: x[0] == 0
        )

    mesh_descr = RectangleMeshDescription(
        initial_position=None,
        max_element_perimeter=0.25 * 10 * mm,
        scale=[8 * 10 * mm, 10 * mm],
    )

    setup = StaticSetup(mesh_descr)

    runner = StaticSolver(setup, solving_method)
    result = runner.solve(initial_displacement=setup.initial_displacement, method=optimization_method)

    displacement = result.body.mesh.nodes[:] - result.displaced_nodes[:]
    std_ids = standard_boundary_nodes(runner.body.mesh.nodes, runner.body.mesh.elements)

    # print result
    np.set_printoptions(precision=8, suppress=True)
    print(repr(displacement[std_ids]))

    np.testing.assert_array_almost_equal(
        displacement[std_ids], expected_displacement_vector, decimal=3
    )
