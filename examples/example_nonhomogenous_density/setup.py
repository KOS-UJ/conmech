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

import numpy as np

from conmech.dynamics.contact.relu_slope_contact_law import make_slope_contact_law
from conmech.mesh.boundaries_description import BoundariesDescription
from conmech.properties.mesh_description import CrossMeshDescription
from conmech.scenarios.problems import StaticDisplacementProblem

OUTPUTS_PATH = "./output/nonhomogenous_density"

MESH_PERIMETER = 0.5
MESH_SCALE = [2.5, 1]
SOLVING_METHOD = "schur"

DENSITY_LEFT = 1
DENSITY_RIGHT = 0.2
DENSITY_SPLIT_X = 1


@dataclass
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


def mesh_description():
    return CrossMeshDescription(
        initial_position=None, max_element_perimeter=MESH_PERIMETER, scale=MESH_SCALE
    )


def element_density(mesh) -> np.ndarray:
    """Piecewise constant density, split along `DENSITY_SPLIT_X`."""
    centers = np.empty(shape=(len(mesh.elements), 2))
    for index, element in enumerate(mesh.elements):
        centers[index] = np.sum(mesh.nodes[element], axis=0) / len(element)
    return np.asarray([DENSITY_LEFT if x[0] < DENSITY_SPLIT_X else DENSITY_RIGHT for x in centers])
