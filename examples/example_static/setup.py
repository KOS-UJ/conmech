# CONMECH @ Jagiellonian University in Kraków
#
# Copyright (C) 2019-2026  Piotr Bartman-Szwarc <piotr.bartman@uj.edu.pl>
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

from conmech.dynamics.contact.relu_slope_contact_law import make_slope_contact_law
from conmech.mesh.boundaries_description import BoundariesDescription
from conmech.properties.mesh_description import (
    CubeMeshDescription,
    RectangleMeshDescription,
)
from conmech.scenarios.problems import StaticDisplacementProblem

OUTPUTS_PATH = "./output/static"

MU_COEF = 4
LA_COEF = 4
CONTACT_SLOPE = 1

MESH_PERIMETER = 0.5
MESH_SCALE = [2.5, 1]


@dataclass
class StaticSetup(StaticDisplacementProblem):
    mu_coef: ... = MU_COEF
    la_coef: ... = LA_COEF
    contact_law: ... = make_slope_contact_law(slope=CONTACT_SLOPE)

    @staticmethod
    def inner_forces(x, t=None):
        return -0.2 * x

    @staticmethod
    def outer_forces(x, t=None):
        return 0 * x

    boundaries: ... = BoundariesDescription(
        contact=lambda x: x[1] == 0, dirichlet=lambda x: x[0] == 0
    )


def mesh_description(dimension: int):
    if dimension == 2:
        return RectangleMeshDescription(
            initial_position=None, max_element_perimeter=MESH_PERIMETER, scale=MESH_SCALE
        )
    return CubeMeshDescription(initial_position=None)


def solving_method(dimension: int) -> str:
    return "schur" if dimension == 2 else "global"
