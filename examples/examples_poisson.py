from dataclasses import dataclass

import numpy as np
from conmech.helpers.config import Config
from conmech.mesh.boundaries_description import BoundariesDescription
from conmech.plotting.drawer import Drawer
from conmech.scenarios.problems import Static
from conmech.simulations.problem_solver import PoissonSolver
from examples.p_slope_contact_law import make_slope_contact_law


@dataclass()
class StaticPoissonSetup(Static):
    grid_height: ... = 1.0
    elements_number: ... = (32, 32)
    mu_coef: ... = 0
    la_coef: ... = 0
    contact_law: ... = make_slope_contact_law(slope=0)

    @staticmethod
    def inner_forces(x):
        return np.array([1000.])

    @staticmethod
    def outer_forces(x):
        return np.array([3.])

    boundaries: ... = BoundariesDescription(
        dirichlet=lambda x: x[0] == 0 or x[0] == 1
    )


def main(show: bool = True, save: bool = False):
    setup = StaticPoissonSetup(mesh_type="cross")
    runner = PoissonSolver(setup, "direct")

    state = runner.solve(verbose=True, initial_displacement=setup.initial_displacement)
    config = Config()
    t_max = max(state.temperature)
    t_min = min(state.temperature)
    Drawer(state=state, config=config).draw(show=show, save=save, temp_max=t_max, temp_min=t_min)


if __name__ == "__main__":
    main(show=True)
