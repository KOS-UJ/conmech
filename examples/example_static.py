"""
Created at 21.08.2019
"""
from dataclasses import dataclass

import numpy as np
from conmech.helpers.config import Config
from conmech.mesh.boundaries_description import BoundariesDescription
from conmech.plotting.drawer import Drawer
from conmech.scenarios.problems import StaticDisplacementProblem
from conmech.simulations.problem_solver import StaticSolver

from examples.p_slope_contact_law import make_slope_contact_law


@dataclass
class StaticSetup(StaticDisplacementProblem):
    grid_height: ... = 1.0
    elements_number: ... = (2, 5)
    mu_coef: ... = 4
    la_coef: ... = 4
    contact_law: ... = make_slope_contact_law(slope=1)

    @staticmethod
    def inner_forces(x):
        return np.array([-0.2, -0.2])

    @staticmethod
    def outer_forces(x):
        return np.array([0, 0])

    @staticmethod
    def friction_bound(u_nu):
        return 0

    boundaries: ... = BoundariesDescription(
        contact=lambda x: x[1] == 0, dirichlet=lambda x: x[0] == 0
    )


def main(show: bool = True, save: bool = False):
    setup = StaticSetup(mesh_type="cross")
    runner = StaticSolver(setup, "schur")

    state = runner.solve(verbose=True, initial_displacement=setup.initial_displacement)
    config = Config()
    Drawer(state=state, config=config).draw(show=show, save=save)


if __name__ == "__main__":
    main(show=True)
