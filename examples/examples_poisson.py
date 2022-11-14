from dataclasses import dataclass

import numpy as np
from conmech.helpers.config import Config
from conmech.mesh.boundaries_description import BoundariesDescription
from conmech.plotting.drawer import Drawer
from conmech.scenarios.problems import Problem
from conmech.simulations.problem_solver import Static as StaticProblemSolver


@dataclass()
class StaticSetup(Problem):
    grid_height: ... = 1.0
    elements_number: ... = (2, 5)

    @staticmethod
    def inner_forces(x):
        return np.array([10.])

    @staticmethod
    def outer_forces(x):
        return np.array([0., 0.])

    boundaries: ... = BoundariesDescription(
        dirichlet=lambda x: x[0] == 0 || x[0] == 1
    )


def main(show: bool = True, save: bool = False):
    setup = StaticSetup(mesh_type="cross")
    runner = StaticProblemSolver(setup, "schur")

    state = runner.solve(verbose=True, initial_displacement=setup.initial_displacement)
    config = Config()
    Drawer(state=state, config=config).draw(show=show, save=save)


if __name__ == "__main__":
    main(show=True)
