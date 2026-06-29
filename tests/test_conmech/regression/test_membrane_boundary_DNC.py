# CONMECH @ Jagiellonian University in Kraków
#
# Copyright (C) 2023  Piotr Bartman-Szwarc <piotr.bartman@uj.edu.pl>
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
from typing import Optional

import numpy as np
import pytest

from conmech.mesh.boundaries_description import BoundariesDescription
from conmech.scenarios.problems import ContactWaveProblem
from conmech.simulations.problem_solver import WaveSolver
from conmech.properties.mesh_description import CrossMeshDescription
from conmech.dynamics.contact.damped_normal_compliance import make_damped_norm_compl


@pytest.fixture(params=["schur", "global optimization"])
def solving_method(request):
    return request.param


def generate_test_suits():
    test_suites = []

    @dataclass()
    class MembraneSetup(ContactWaveProblem):
        time_step: ... = 0.1
        propagation: ... = 1.0
        contact_law: ... = make_damped_norm_compl(0.01, kappa=1.0, beta=0.5)()

        @staticmethod
        def inner_forces(x: np.ndarray, t: Optional[float] = None) -> np.ndarray:
            return np.array([100])

        @staticmethod
        def outer_forces(
            x: np.ndarray, v: Optional[np.ndarray] = None, t: Optional[float] = None
        ) -> np.ndarray:
            return np.array([0.0])

        boundaries: ... = BoundariesDescription(
            dirichlet=lambda x: x[0] in (0,) or x[1] in (0, 1),
            contact=lambda x: x[0] == 1,
        )

    mesh_descr = CrossMeshDescription(
        initial_position=None, max_element_perimeter=1 / 6, scale=[1, 1]
    )
    setup_1 = MembraneSetup(mesh_descr)

    # setup0
    expected_displacement_vector_1 = [
        [-1.4461375, 0.],
        [-1.78864151, 0.],
        [-1.84441797, 0.],
        [-1.78864395, 0.],
        [-1.44613752, 0.],
        [-1.05870528, 0.],
        [-1.21740155, 0.],
        [-1.24884761, 0.],
        [-1.23629703, 0.],
        [-1.13532425, 0.],
        [-0.65005644, 0.],
        [-2.06033899, 0.],
        [-2.43882905, 0.],
        [-2.51873194, 0.],
        [-2.48580691, 0.],
        [-2.23271557, 0.],
        [-1.13503764, 0.],
        [-2.28571348, 0.],
        [-2.73744479, 0.],
        [-2.83858208, 0.],
        [-2.7954914, 0.],
        [-2.48021767, 0.],
        [-1.23382591, 0.],
        [-2.28571299, 0.],
        [-2.73744387, 0.],
        [-2.83858207, 0.],
        [-2.79549114, 0.],
        [-2.48021831, 0.],
        [-1.23382438, 0.],
        [-2.06033918, 0.],
        [-2.43882792, 0.],
        [-2.51873344, 0.],
        [-2.48580782, 0.],
        [-2.23271679, 0.],
        [-1.13504064, 0.],
        [-1.05870559, 0.],
        [-1.21740214, 0.],
        [-1.24884847, 0.],
        [-1.23629817, 0.],
        [-1.13532837, 0.],
        [-0.65005682, 0.],
        [-1.98863762, 0.],
        [-2.12051382, 0.],
        [-2.13304394, 0.],
        [-2.06104677, 0.],
        [-1.66772019, 0.],
        [-2.53792336, 0.],
        [-2.74039086, 0.],
        [-2.76047592, 0.],
        [-2.64385136, 0.],
        [-2.05965383, 0.],
        [-2.63304919, 0.],
        [-2.85312687, 0.],
        [-2.87533778, 0.],
        [-2.74551785, 0.],
        [-2.12254481, 0.],
        [-2.53792388, 0.],
        [-2.74039033, 0.],
        [-2.76047724, 0.],
        [-2.64385116, 0.],
        [-2.05965484, 0.],
        [-1.98863637, 0.],
        [-2.12051392, 0.],
        [-2.13304381, 0.],
        [-2.06104804, 0.],
        [-1.66772345, 0.],
        [0., 0.],
        [0., 0.],
        [0., 0.],
        [0., 0.],
        [0., 0.],
        [0., 0.],
        [0., 0.],
        [0., 0.],
        [0., 0.],
        [0., 0.],
        [0., 0.],
        [0., 0.],
        [0., 0.],
        [0., 0.],
        [0., 0.],
        [0., 0.],
        [0., 0.],
        [0., 0.],
        [0., 0.],
    ]

    test_suites.append((setup_1, expected_displacement_vector_1))

    # various changes

    @dataclass()
    class MembraneSetup(ContactWaveProblem):
        time_step: ... = 0.1
        propagation: ... = 1.0
        contact_law: ... = make_damped_norm_compl(0.01, kappa=10.0, beta=0.5)()

        @staticmethod
        def inner_forces(x: np.ndarray, t: Optional[float] = None) -> np.ndarray:
            return np.array([-100])

        @staticmethod
        def outer_forces(
            x: np.ndarray, v: Optional[np.ndarray] = None, t: Optional[float] = None
        ) -> np.ndarray:
            return np.array([0.0])

        boundaries: ... = BoundariesDescription(
            dirichlet=lambda x: x[0] in (0,) or x[1] in (0, 1),
            contact=lambda x: x[0] == 1,
        )

    mesh_descr = CrossMeshDescription(
        initial_position=None, max_element_perimeter=1 / 4, scale=[1, 1]
    )
    setup_2 = MembraneSetup(mesh_descr)

    # setup1
    expected_displacement_vector_2 = [
        [2.64317675, 0.0],
        [2.96235934, 0.0],
        [2.64317508, 0.0],
        [1.67732622, 0.0],
        [1.67610052, 0.0],
        [1.63886416, 0.0],
        [1.09339295, 0.0],
        [2.87365321, 0.0],
        [2.87056248, 0.0],
        [2.77947225, 0.0],
        [1.63768291, 0.0],
        [2.87365323, 0.0],
        [2.87056375, 0.0],
        [2.77947193, 0.0],
        [1.63768443, 0.0],
        [1.67732504, 0.0],
        [1.67610041, 0.0],
        [1.63886298, 0.0],
        [1.09339442, 0.0],
        [2.64291874, 0.0],
        [2.63536374, 0.0],
        [2.40190544, 0.0],
        [2.96209659, 0.0],
        [2.94988312, 0.0],
        [2.62782706, 0.0],
        [2.64291756, 0.0],
        [2.63536304, 0.0],
        [2.40190459, 0.0],
        [0.0, 0.0],
        [0.0, 0.0],
        [0.0, 0.0],
        [0.0, 0.0],
        [0.0, 0.0],
        [0.0, 0.0],
        [0.0, 0.0],
        [0.0, 0.0],
        [0.0, 0.0],
        [0.0, 0.0],
        [0.0, 0.0],
        [0.0, 0.0],
        [0.0, 0.0],
    ]

    test_suites.append((setup_2, expected_displacement_vector_2))

    return test_suites


@pytest.mark.parametrize("setup, expected_displacement_vector", generate_test_suits())
def test_boundary_DNC(solving_method, setup, expected_displacement_vector):
    runner = WaveSolver(setup, solving_method)
    results = runner.solve(
        n_steps=2,
        initial_displacement=setup.initial_displacement,
        initial_velocity=setup.initial_velocity,
    )

    displacement = results[-1].body.mesh.nodes[:] - results[-1].displaced_nodes[:]

    # print result
    np.set_printoptions(precision=8, suppress=True)
    print(repr(displacement))

    precision = 2 if solving_method != "global optimization" else 3
    np.testing.assert_array_almost_equal(displacement, expected_displacement_vector, decimal=2)
