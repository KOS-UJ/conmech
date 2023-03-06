# CONMECH @ Jagiellonian University in Kraków
#
# Copyright (C) 2023  Piotr Bartman <piotr.bartman@uj.edu.pl>
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
from conmech.scenarios.problems import LongMemoryQuasistaticProblem
from conmech.simulations.problem_solver import TimeDependentLongMemory

from examples.p_slope_contact_law import make_slope_contact_law


eps = 1e-18

@dataclass()
class QuasistaticSetup(LongMemoryQuasistaticProblem):
    grid_height: ... = 1.0
    elements_number: ... = (2, 5)
    mu_coef: ... = 40
    la_coef: ... = 40
    th_coef: ... = 400
    ze_coef: ... = 400
    time_step: ... = 0.1
    contact_law: ... = make_slope_contact_law(slope=2)

    @staticmethod
    def inner_forces(x):
        return np.array([0.0, 0.0])

    @staticmethod
    def outer_forces(x):
        return np.array([-0.2, -0.2])

    @staticmethod
    def friction_bound(u_nu):
        return 0

    boundaries: ... = BoundariesDescription(
        contact=lambda x: x[0] >= 4 and x[1] < eps, dirichlet=lambda x: x[0] <= 1 and x[1] < eps
    )


def main(show: bool = True, save: bool = False):
    setup = QuasistaticSetup(mesh_type="tunnel")
    runner = TimeDependentLongMemory(setup, solving_method="schur")

    states = runner.solve(
        n_steps=32,
        output_step=(0, 32),
        verbose=True,
        initial_absement=setup.initial_absement,
        initial_displacement=setup.initial_displacement,
    )
    config = Config()
    for state in states:
        Drawer(state=state, config=config).draw(show=show, save=save)


if __name__ == "__main__":
    main(show=True)
