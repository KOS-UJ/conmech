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
from conmech.properties.mesh_description import ImportedMeshDescription
from conmech.scenarios.problems import StaticDisplacementProblem

OUTPUTS_PATH = "./output/imported_mesh_static"

E = 10000
kappa = 0.4

MESH_PATH = "meshes/example_mesh.msh"
SOLVING_METHOD = "schur"


@dataclass
class StaticSetup(StaticDisplacementProblem):
    mu_coef: ... = E / (1 + kappa)
    la_coef: ... = E * kappa / ((1 + kappa) * (1 - 2 * kappa))
    contact_law: ... = make_slope_contact_law(slope=1)

    @staticmethod
    def inner_forces(x, t=None):
        return np.array([0, 0])

    @staticmethod
    def outer_forces(x, t=None):
        return np.array([0, -1]) if x[0] > 1.9 and x[1] < 0.1 else np.zeros(2)

    boundaries: ... = BoundariesDescription(
        contact=lambda x: x[1] == 0 and x[0] < 0.5, dirichlet=lambda x: x[0] == 0
    )


def mesh_description(mesh_path: str = MESH_PATH):
    return ImportedMeshDescription(initial_position=None, path=mesh_path)
