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
    def inner_forces(x, t=None):
        return np.array([-0.2, -0.2])

    @staticmethod
    def outer_forces(x, t=None):
        return np.array([0, 0])

    @staticmethod
    def friction_bound(u_nu: float) -> float:
        return 0

    boundaries: ... = BoundariesDescription(
        contact=lambda x: x[1] == 0, dirichlet=lambda x: x[0] == 0
    )


def main(config: Config):
    """
    Entrypoint to example.

    To see result of simulation you need to call from python `main(Config().init())`.
    """
    setup = StaticSetup(mesh_type="cross")
    runner = StaticSolver(setup, "schur")

    state = runner.solve(verbose=True, initial_displacement=setup.initial_displacement)
    Drawer(state=state, config=config).draw(show=config.show, save=config.save)


if __name__ == "__main__":
    main(Config().init())
