from dataclasses import dataclass

import numpy as np
from conmech.helpers.config import Config
from conmech.mesh.boundaries_description import BoundariesDescription
from conmech.plotting.drawer import Drawer
from conmech.scenarios.problems import PoissonProblem
from conmech.simulations.problem_solver import PoissonSolver


@dataclass()
class StaticPoissonSetup(PoissonProblem):
    grid_height: ... = 1
    elements_number: ... = (14, 14)

    @staticmethod
    def inner_forces(x: np.ndarray) -> np.ndarray:
        return np.array([0.0])

    @staticmethod
    def outer_forces(x: np.ndarray) -> np.ndarray:
        return np.array([10.0])

    boundaries: ... = BoundariesDescription(dirichlet=lambda x: x[0] == 0 or x[0] == 1)


def main(show: bool = True, save: bool = False):
    setup = StaticPoissonSetup(mesh_type="meshzoo")
    runner = PoissonSolver(setup, "direct")

    state = runner.solve(verbose=True)
    config = Config()
    t_max = max(state.temperature)
    t_min = min(state.temperature)
    Drawer(state=state, config=config).draw(show=show, save=save, temp_max=t_max, temp_min=t_min)


if __name__ == "__main__":
    main(show=True)
