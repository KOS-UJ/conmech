# CONMECH @ Jagiellonian University in Kraków
#
# Copyright (C) 2025-2026  Piotr Bartman-Szwarc <piotr.bartman@uj.edu.pl>
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
from typing import Optional, Type

import numpy as np

from conmech.dynamics.contact.contact_law import ContactLaw, PotentialOfContactLaw
from conmech.mesh.boundaries_description import BoundariesDescription
from conmech.properties.mesh_description import RectangleMeshDescription
from conmech.scenarios.problems import PoissonProblem

OUTPUTS_PATH = "./output/BOT2023"

ALPHAS = [0.01, 0.1, 1, 10, 100, 1000, 10_000, 100_000, 1_000_000, np.inf]
IHS = [4, 8, 16, 32, 48, 72]
B_COEF = 5
MAXD = 72
TEMPERATURE_GRID = (
    ((np.inf, 4), (np.inf, MAXD)),
    ((10, MAXD), (100, MAXD)),
    ((0.1, MAXD), (1, MAXD)),
    ((0.01, 4), (0.01, MAXD)),
)
CONVERGENCE_SEQUENCES = (
    (
        tuple((0.01, h) for h in IHS),
        tuple((np.inf, h) for h in IHS),
    ),
    (
        tuple((a, 4) for a in ALPHAS),
        tuple((a, 72) for a in ALPHAS),
    ),
    (
        (
            (0.01, 4),
            (0.1, 8),
            (1, 8),
            (10, 16),
            (100, 16),
            (1000, 32),
            (10_000, 32),
            (100_000, 48),
            (1_000_000, 72),
            (np.inf, 72),
        ),
        None,
    ),
)


def make_slope_contact_law(slope: float) -> Type[ContactLaw]:
    class TarziaContactLaw(PotentialOfContactLaw):
        @staticmethod
        def potential_normal_direction(
            var_nu: float, static_displacement_nu: float, dt: float
        ) -> float:
            b = B_COEF
            r = var_nu
            # EXAMPLE 11
            if r < b:
                result = (r - b) ** 2
            else:
                result = 2 * np.log((r - b) + 1)
            result *= slope
            return result

        @staticmethod
        def subderivative_normal_direction(
            var_nu: float, static_displacement_nu: float, dt: float
        ) -> float:
            b = B_COEF
            r = var_nu
            # EXAMPLE 11
            if r < b:
                result = r - b
            else:
                result = 1 / (r - b + 1)
            result *= slope
            return result

    return TarziaContactLaw


@dataclass()
class StaticPoissonSetup(PoissonProblem):
    contact_law_2: Type[ContactLaw] = make_slope_contact_law(slope=1000)

    @staticmethod
    def internal_temperature(x: np.ndarray, t: Optional[float] = None) -> np.ndarray:
        return np.array([-4])

    @staticmethod
    def outer_temperature(x: np.ndarray, t: Optional[float] = None) -> np.ndarray:
        _y = x[1]
        return np.array([_y * (_y - 1) * 32])

    boundaries: BoundariesDescription = BoundariesDescription(
        dirichlet=(
            lambda x: x[1] == 0.0,  # or x[1] == 1.0,
            lambda x: np.full(x.shape[0], 5),
        ),
        contact=lambda x: x[1] == 1.0,
    )


def mesh_description(ih: int):
    return RectangleMeshDescription(
        initial_position=None, max_element_perimeter=1 / ih, scale=[2, 1]
    )


def solving_method(alpha) -> str:
    return "schur" if alpha != np.inf else "direct"
