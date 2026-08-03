# CONMECH @ Jagiellonian University in Kraków
#
# Copyright (C) 2022-2026  Piotr Bartman <piotr.bartman@uj.edu.pl>
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
from conmech.scenarios.problems import PoissonProblem

OUTPUTS_PATH = "./output/poisson"

MESH_PERIMETER = 0.125
MESH_SCALE = [1, 1]
SOLVING_METHOD = "direct"


@dataclass()
class StaticPoissonSetup(PoissonProblem):
    @staticmethod
    def internal_temperature(x: np.ndarray, t: Optional[float] = None) -> np.ndarray:
        if 0.4 <= x[0] <= 0.6 and 0.4 <= x[1] <= 0.6:
            return np.array([-10.0])
        return np.array([2 * np.pi**2 * np.sin(np.pi * x[0]) * np.sin(np.pi * x[1])])

    @staticmethod
    def outer_temperature(
        x: np.ndarray, v: Optional[np.ndarray] = None, t: Optional[float] = None
    ) -> np.ndarray:
        if x[0] == 1:
            return np.array([10.0])
        return np.array([0.0])

    boundaries: ... = BoundariesDescription(
        dirichlet=(
            lambda x: x[1] == 0 or x[0] == 0 or x[1] == 1,
            lambda x: np.full(x.shape[0], 0),
        )
    )


def mesh_description():
    return CrossMeshDescription(
        initial_position=None, max_element_perimeter=MESH_PERIMETER, scale=MESH_SCALE
    )
