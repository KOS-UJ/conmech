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
from conmech.scenarios.problems import WaveProblem
from conmech.simulations.problem_solver import WaveSolver
from conmech.properties.mesh_description import CrossMeshDescription


@pytest.fixture(params=["direct"])
def solving_method(request):
    return request.param


def generate_test_suits():
    test_suites = []

    @dataclass()
    class MembraneSetup(WaveProblem):
        time_step: ... = 0.1
        propagation: ... = 1.0

        @staticmethod
        def inner_forces(x: np.ndarray, t: Optional[float] = None) -> np.ndarray:
            return np.array([2.0])

        @staticmethod
        def outer_forces(
            x: np.ndarray, v: Optional[np.ndarray] = None, t: Optional[float] = None
        ) -> np.ndarray:
            return np.array([0.0])

        boundaries: ... = BoundariesDescription(
            dirichlet=lambda x: x[0] in (0, 1) or x[1] in (0, 1)
        )

    mesh_descr = CrossMeshDescription(
        initial_position=None, max_element_perimeter=0.125, scale=[1, 1]
    )
    setup_1 = MembraneSetup(mesh_descr)

    expected_displacement_vector_1 = [
        [-0.01001162, 0.0],
        [-0.0161334, 0.0],
        [-0.01735278, 0.0],
        [-0.01761153, 0.0],
        [-0.01761153, 0.0],
        [-0.01735278, 0.0],
        [-0.0161334, 0.0],
        [-0.01001162, 0.0],
        [-0.0161334, 0.0],
        [-0.02990235, 0.0],
        [-0.03301072, 0.0],
        [-0.03370103, 0.0],
        [-0.03370103, 0.0],
        [-0.03301072, 0.0],
        [-0.02990235, 0.0],
        [-0.0161334, 0.0],
        [-0.01735278, 0.0],
        [-0.03301072, 0.0],
        [-0.03699519, 0.0],
        [-0.03793492, 0.0],
        [-0.03793492, 0.0],
        [-0.03699519, 0.0],
        [-0.03301072, 0.0],
        [-0.01735278, 0.0],
        [-0.01761153, 0.0],
        [-0.03370103, 0.0],
        [-0.03793492, 0.0],
        [-0.03896977, 0.0],
        [-0.03896977, 0.0],
        [-0.03793492, 0.0],
        [-0.03370103, 0.0],
        [-0.01761153, 0.0],
        [-0.01761153, 0.0],
        [-0.03370103, 0.0],
        [-0.03793492, 0.0],
        [-0.03896977, 0.0],
        [-0.03896977, 0.0],
        [-0.03793492, 0.0],
        [-0.03370103, 0.0],
        [-0.01761153, 0.0],
        [-0.01735278, 0.0],
        [-0.03301072, 0.0],
        [-0.03699519, 0.0],
        [-0.03793492, 0.0],
        [-0.03793492, 0.0],
        [-0.03699519, 0.0],
        [-0.03301072, 0.0],
        [-0.01735278, 0.0],
        [-0.0161334, 0.0],
        [-0.02990235, 0.0],
        [-0.03301072, 0.0],
        [-0.03370103, 0.0],
        [-0.03370103, 0.0],
        [-0.03301072, 0.0],
        [-0.02990235, 0.0],
        [-0.0161334, 0.0],
        [-0.01001162, 0.0],
        [-0.0161334, 0.0],
        [-0.01735278, 0.0],
        [-0.01761153, 0.0],
        [-0.01761153, 0.0],
        [-0.01735278, 0.0],
        [-0.0161334, 0.0],
        [-0.01001162, 0.0],
        [-0.02333987, 0.0],
        [-0.02789758, 0.0],
        [-0.02889672, 0.0],
        [-0.02907672, 0.0],
        [-0.02889672, 0.0],
        [-0.02789758, 0.0],
        [-0.02333987, 0.0],
        [-0.02789758, 0.0],
        [-0.03484893, 0.0],
        [-0.0365059, 0.0],
        [-0.03681558, 0.0],
        [-0.0365059, 0.0],
        [-0.03484893, 0.0],
        [-0.02789758, 0.0],
        [-0.02889672, 0.0],
        [-0.0365059, 0.0],
        [-0.03844603, 0.0],
        [-0.03882171, 0.0],
        [-0.03844603, 0.0],
        [-0.0365059, 0.0],
        [-0.02889672, 0.0],
        [-0.02907672, 0.0],
        [-0.03681558, 0.0],
        [-0.03882171, 0.0],
        [-0.03921568, 0.0],
        [-0.03882171, 0.0],
        [-0.03681558, 0.0],
        [-0.02907672, 0.0],
        [-0.02889672, 0.0],
        [-0.0365059, 0.0],
        [-0.03844603, 0.0],
        [-0.03882171, 0.0],
        [-0.03844603, 0.0],
        [-0.0365059, 0.0],
        [-0.02889672, 0.0],
        [-0.02789758, 0.0],
        [-0.03484893, 0.0],
        [-0.0365059, 0.0],
        [-0.03681558, 0.0],
        [-0.0365059, 0.0],
        [-0.03484893, 0.0],
        [-0.02789758, 0.0],
        [-0.02333987, 0.0],
        [-0.02789758, 0.0],
        [-0.02889672, 0.0],
        [-0.02907672, 0.0],
        [-0.02889672, 0.0],
        [-0.02789758, 0.0],
        [-0.02333987, 0.0],
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
        [0.0, 0.0],
        [0.0, 0.0],
        [0.0, 0.0],
        [0.0, 0.0],
        [0.0, 0.0],
        [0.0, 0.0],
    ]

    test_suites.append((setup_1, expected_displacement_vector_1))

    # various changes

    @dataclass()
    class MembraneSetup(WaveProblem):
        time_step: ... = 0.1
        propagation: ... = 1.0

        @staticmethod
        def inner_forces(x: np.ndarray, t: Optional[float] = None) -> np.ndarray:
            return np.array([0.0])

        @staticmethod
        def outer_forces(
            x: np.ndarray, v: Optional[np.ndarray] = None, t: Optional[float] = None
        ) -> np.ndarray:
            return np.array([1.0])

        boundaries: ... = BoundariesDescription(
            dirichlet=lambda x: x[0] in (0, 1) or x[1] in (0, 1)
        )

    setup_2 = MembraneSetup(mesh_descr)

    expected_displacement_vector_2 = np.zeros((145, 2))

    test_suites.append((setup_2, expected_displacement_vector_2))

    return test_suites


@pytest.mark.parametrize("setup, expected_displacement_vector", generate_test_suits())
def test_dynamic_membrane(solving_method, setup, expected_displacement_vector):
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

    np.testing.assert_array_almost_equal(
        displacement, expected_displacement_vector, decimal=3
    )
