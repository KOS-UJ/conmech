# CONMECH @ Jagiellonian University in Kraków
#
# Copyright (C) 2023-2026  Piotr Bartman-Szwarc <piotr.bartman@uj.edu.pl>
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

from conmech.mesh.boundaries_description import BoundariesDescription
from conmech.properties.mesh_description import CrossMeshDescription
from conmech.scenarios.problems import WaveProblem

OUTPUTS_PATH = "./output/dynamic_membrane"

MESH_PERIMETER = 1 / 8
MESH_PERIMETER_TEST = 1 / 3
MESH_SCALE = [1, 1]
SOLVING_METHOD = "direct"
N_STEPS = 32
N_STEPS_TEST = 3


@dataclass()
class MembraneSetup(WaveProblem):
    time_step: ... = 0.1
    propagation: ... = 1.0

    @staticmethod
    def inner_forces(x: np.ndarray, t: Optional[float] = None) -> np.ndarray:
        return np.array([0.2])

    @staticmethod
    def outer_forces(
        x: np.ndarray, v: Optional[np.ndarray] = None, t: Optional[float] = None
    ) -> np.ndarray:
        return np.array([0.0])

    boundaries: ... = BoundariesDescription(dirichlet=lambda x: x[0] in (0, 1) or x[1] in (0, 1))


def mesh_description(test: bool = False):
    return CrossMeshDescription(
        initial_position=None,
        max_element_perimeter=MESH_PERIMETER_TEST if test else MESH_PERIMETER,
        scale=MESH_SCALE,
    )
