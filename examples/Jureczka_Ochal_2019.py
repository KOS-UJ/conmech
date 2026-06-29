# CONMECH @ Jagiellonian University in Kraków
#
# Copyright (C) 2024  Piotr Bartman-Szwarc <piotr.bartman@uj.edu.pl>
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
from conmech.helpers.config import Config
from conmech.mesh.boundaries_description import BoundariesDescription
from conmech.plotting.drawer import Drawer
from conmech.scenarios.problems import StaticDisplacementProblem
from conmech.dynamics.contact.contact_law import PotentialOfContactLaw
from conmech.simulations.problem_solver import StaticSolver as StaticProblemSolver
from conmech.properties.mesh_description import CrossMeshDescription


class JureczkaOchal2019(PotentialOfContactLaw):
    @staticmethod
    def potential_normal_direction(
        var_nu: float, static_displacement_nu: float, dt: float
    ) -> float:
        if var_nu <= 0:
            return 0.0
        if var_nu < 0.1:
            return 10 * var_nu * var_nu
        return 0.1

    @staticmethod
    def potential_tangential_direction(
        var_tau: float, static_displacement_tau: float, dt: float
    ) -> float:
        return np.log(np.sum(var_tau * var_tau) ** 0.5 + 1)

    @staticmethod
    def tangential_bound(var_nu: float, static_displacement_nu: float, dt: float) -> float:
        if static_displacement_nu < 0:
            return 0
        if static_displacement_nu < 0.1:
            return 8 * static_displacement_nu
        return 0.8


@dataclass
class StaticSetup(StaticDisplacementProblem):
    mu_coef: ... = 4
    la_coef: ... = 4
    contact_law: ... = JureczkaOchal2019

    @staticmethod
    def inner_forces(x, t=None):
        return np.array([-1.2, -0.9])

    @staticmethod
    def outer_forces(x, t=None):
        return np.array([0, 0])

    boundaries: ... = BoundariesDescription(
        contact=lambda x: x[1] == 0, dirichlet=lambda x: x[0] == 0
    )


def main(config: Config):
    """
    Entrypoint to example.

    To see result of simulation you need to call from python `main(Config().init())`.
    """
    mesh_descr = CrossMeshDescription(
        initial_position=None, max_element_perimeter=1 / 16, scale=[2, 1]
    )
    setup = StaticSetup(mesh_descr)
    if config.test:
        setup.elements_number = (2, 4)
    runner = StaticProblemSolver(setup, "schur")

    state = runner.solve(
        verbose=True,
        fixed_point_abs_tol=0.001,
        initial_displacement=setup.initial_displacement,
    )
    Drawer(state=state, config=config).draw(show=config.show, save=config.save)


if __name__ == "__main__":
    main(Config().init())
