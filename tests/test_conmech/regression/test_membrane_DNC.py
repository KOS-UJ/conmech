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
from conmech.scenarios.problems import InteriorContactWaveProblem
from conmech.simulations.problem_solver import WaveSolver
from conmech.properties.mesh_description import CrossMeshDescription
from examples.BOSK_2024_example_2 import make_DNC


@pytest.fixture(params=["global"])
def solving_method(request):
    return request.param


def generate_test_suits():
    test_suites = []

    @dataclass()
    class MembraneSetup(InteriorContactWaveProblem):
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
            dirichlet=lambda x: x[0] in (0, 1) or x[1] in (0, 1)
        )

    mesh_descr = CrossMeshDescription(
        initial_position=None, max_element_perimeter=1/4, scale=[1, 1]
    )
    setup_1 = MembraneSetup(mesh_descr)

    expected_displacement_vector_1 = [
        [-1.0900725, -0.],
        [-1.6326846, -0.],
        [-1.6326838, -0.],
        [-1.0900737, -0.],
        [-1.6326846, -0.],
        [-2.7679646, -0.],
        [-2.767965, -0.],
        [-1.6326848, -0.],
        [-1.6326846, -0.],
        [-2.7679654, -0.],
        [-2.7679641, -0.],
        [-1.6326849, -0.],
        [-1.0900731, -0.],
        [-1.6326842, -0.],
        [-1.6326833, -0.],
        [-1.0900724, -0.],
        [-2.3944331, -0.],
        [-2.6196592, -0.],
        [-2.3944336, -0.],
        [-2.6196581, -0.],
        [-2.928575, -0.],
        [-2.6196588, -0.],
        [-2.3944323, -0.],
        [-2.6196583, -0.],
        [-2.3944328, -0.],
        [0., -0.],
        [0., -0.],
        [0., -0.],
        [0., -0.],
        [0., -0.],
        [0., -0.],
        [0., -0.],
        [0., -0.],
        [0., -0.],
        [0., -0.],
        [0., -0.],
        [0., -0.],
        [0., -0.],
        [0., -0.],
        [0., -0.],
        [0., -0.],
    ]

    test_suites.append((setup_1, expected_displacement_vector_1))

    # various changes

    @dataclass()
    class MembraneSetup(InteriorContactWaveProblem):
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
            dirichlet=lambda x: x[0] in (0, 1) or x[1] in (0, 1)
        )

    setup_2 = MembraneSetup(mesh_descr)

    expected_displacement_vector_2 = [
        [1.0933539, -0.],
        [1.6375976, -0.],
        [1.6375984, -0.],
        [1.0933542, -0.],
        [1.6375979, -0.],
        [2.7762924, -0.],
        [2.7762933, -0.],
        [1.6375972, -0.],
        [1.6375985, -0.],
        [2.7762936, -0.],
        [2.7762937, -0.],
        [1.6375977, -0.],
        [1.0933532, -0.],
        [1.6375972, -0.],
        [1.6375986, -0.],
        [1.0933544, -0.],
        [2.401639, -0.],
        [2.6275417, -0.],
        [2.4016391, -0.],
        [2.6275408, -0.],
        [2.9373864, -0.],
        [2.627541, -0.],
        [2.4016392, -0.],
        [2.6275426, -0.],
        [2.4016395, -0.],
        [0., -0.],
        [0., -0.],
        [0., -0.],
        [0., -0.],
        [0., -0.],
        [0., -0.],
        [0., -0.],
        [0., -0.],
        [0., -0.],
        [0., -0.],
        [0., -0.],
        [0., -0.],
        [0., -0.],
        [0., -0.],
        [0., -0.],
        [0., -0.],
    ]

    test_suites.append((setup_2, expected_displacement_vector_2))

    return test_suites


@pytest.mark.parametrize("setup, expected_displacement_vector",
                         generate_test_suits())
def test_membrane_DNC(solving_method, setup, expected_displacement_vector):
    runner = WaveSolver(setup, solving_method)
    results = runner.solve(
        n_steps=2,
        initial_displacement=setup.initial_displacement,
        initial_velocity=setup.initial_velocity,
    )

    displacement = results[-1].body.mesh.nodes[:] - results[-1].displaced_nodes[:]

    # print result
    np.set_printoptions(precision=7, suppress=True)
    print(repr(displacement))

    np.testing.assert_array_almost_equal(
        displacement, expected_displacement_vector, decimal=2
    )
