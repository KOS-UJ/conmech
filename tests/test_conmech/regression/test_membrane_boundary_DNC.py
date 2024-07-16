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
from examples.BOSK_2024_example_1 import make_DNC


@pytest.fixture(params=["schur", "global optimization"])
def solving_method(request):
    return request.param


def generate_test_suits():
    test_suites = []

    @dataclass()
    class MembraneSetup(ContactWaveProblem):
        time_step: ... = 0.1
        propagation: ... = 1.0
        contact_law: ... = make_DNC(0.01, kappa=1.0, beta=0.5)()

        @staticmethod
        def inner_forces(
                x: np.ndarray,
                t: Optional[float] = None
        ) -> np.ndarray:
            return np.array([100])

        @staticmethod
        def outer_forces(
                x: np.ndarray, v: Optional[np.ndarray] = None,
                t: Optional[float] = None
        ) -> np.ndarray:
            return np.array([0.0])

        boundaries: ... = BoundariesDescription(
            dirichlet=lambda x: x[0] in (0,) or x[1] in (0, 1),
            contact=lambda x: x[0] == 1,
        )

    mesh_descr = CrossMeshDescription(
        initial_position=None, max_element_perimeter=1/6, scale=[1, 1]
    )
    setup_1 = MembraneSetup(mesh_descr)

    expected_displacement_vector_1 = [
        [-1.50875561, 0.],
        [-1.80948135, 0.],
        [-1.85151694, 0.],
        [-1.80948058, 0.],
        [-1.50875435, 0.],
        [-1.07643442, 0.],
        [-1.21991845, 0.],
        [-1.24919285, 0.],
        [-1.23634646, 0.],
        [-1.13533086, 0.],
        [-0.65005518, 0.],
        [-2.08407485, 0.],
        [-2.4433707, 0.],
        [-2.5194924, 0.],
        [-2.48593163, 0.],
        [-2.23273285, 0.],
        [-1.13503756, 0.],
        [-2.29408448, 0.],
        [-2.74062776, 0.],
        [-2.83936081, 0.],
        [-2.79564395, 0.],
        [-2.48024186, 0.],
        [-1.2338268, 0.],
        [-2.29408387, 0.],
        [-2.74062736, 0.],
        [-2.83936113, 0.],
        [-2.79564248, 0.],
        [-2.48024111, 0.],
        [-1.2338265, 0.],
        [-2.08407363, 0.],
        [-2.44337049, 0.],
        [-2.51949141, 0.],
        [-2.48593195, 0.],
        [-2.23273261, 0.],
        [-1.13503913, 0.],
        [-1.07643343, 0.],
        [-1.21991743, 0.],
        [-1.24919236, 0.],
        [-1.23634806, 0.],
        [-1.13533196, 0.],
        [-0.65005674, 0.],
        [-1.99885568, 0.],
        [-2.12194468, 0.],
        [-2.13325662, 0.],
        [-2.06107777, 0.],
        [-1.66772572, 0.],
        [-2.54556707, 0.],
        [-2.74205763, 0.],
        [-2.76078498, 0.],
        [-2.64390479, 0.],
        [-2.05966075, 0.],
        [-2.63717937, 0.],
        [-2.85454336, 0.],
        [-2.87565865, 0.],
        [-2.74557381, 0.],
        [-2.12255092, 0.],
        [-2.54556523, 0.],
        [-2.74205761, 0.],
        [-2.76078459, 0.],
        [-2.64390367, 0.],
        [-2.05966124, 0.],
        [-1.99885497, 0.],
        [-2.12194321, 0.],
        [-2.13325616, 0.],
        [-2.0610787, 0.],
        [-1.66772549, 0.],
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
        contact_law: ... = make_DNC(0.01, kappa=10.0, beta=0.5)()

        @staticmethod
        def inner_forces(
                x: np.ndarray,
                t: Optional[float] = None
        ) -> np.ndarray:
            return np.array([-100])

        @staticmethod
        def outer_forces(
                x: np.ndarray, v: Optional[np.ndarray] = None,
                t: Optional[float] = None
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

    expected_displacement_vector_2 = [
        [2.64317675, 0.],
        [2.96235934, 0.],
        [2.64317508, 0.],
        [1.67732622, 0.],
        [1.67610052, 0.],
        [1.63886416, 0.],
        [1.09339295, 0.],
        [2.87365321, 0.],
        [2.87056248, 0.],
        [2.77947225, 0.],
        [1.63768291, 0.],
        [2.87365323, 0.],
        [2.87056375, 0.],
        [2.77947193, 0.],
        [1.63768443, 0.],
        [1.67732504, 0.],
        [1.67610041, 0.],
        [1.63886298, 0.],
        [1.09339442, 0.],
        [2.64291874, 0.],
        [2.63536374, 0.],
        [2.40190544, 0.],
        [2.96209659, 0.],
        [2.94988312, 0.],
        [2.62782706, 0.],
        [2.64291756, 0.],
        [2.63536304, 0.],
        [2.40190459, 0.],
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

    test_suites.append((setup_2, expected_displacement_vector_2))

    return test_suites


@pytest.mark.parametrize("setup, expected_displacement_vector",
                         generate_test_suits())
def test_boundary_DNC(solving_method, setup, expected_displacement_vector):
    runner = WaveSolver(setup, solving_method)
    results = runner.solve(
        n_steps=2,
        initial_displacement=setup.initial_displacement,
        initial_velocity=setup.initial_velocity,
    )

    displacement = results[-1].body.mesh.nodes[:] - results[-1].displaced_nodes[
                                                    :]

    # print result
    np.set_printoptions(precision=8, suppress=True)
    print(repr(displacement))

    precision = 2 if solving_method != "global optimization" else 3
    np.testing.assert_array_almost_equal(
        displacement, expected_displacement_vector, decimal=2
    )
