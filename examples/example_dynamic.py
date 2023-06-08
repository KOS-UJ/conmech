"""
Created at 21.08.2019
"""
from dataclasses import dataclass

import numpy as np

from conmech.helpers.config import Config
from conmech.mesh.boundaries_description import BoundariesDescription
from conmech.plotting.drawer import Drawer
from conmech.scenarios.problems import Dynamic
from conmech.simulations.problem_solver import TimeDependent as TimeDependentProblemSolver
from examples.p_slope_contact_law import make_slope_contact_law


@dataclass()
class DynamicSetup(Dynamic):
    grid_height: ... = 1.0
    elements_number: ... = (2, 5)
    boundaries: ... = BoundariesDescription(
        contact=lambda x: x[1] == 0, dirichlet=lambda x: x[0] == 0
    )
    mu_coef: ... = 4
    la_coef: ... = 4
    th_coef: ... = 4
    ze_coef: ... = 4
    time_step: ... = 0.1
    contact_law: ... = make_slope_contact_law(slope=1e1)

    @staticmethod
    def inner_forces(x, t=None):
        return np.array([-0.2, -0.2])

    @staticmethod
    def outer_forces(x, t=None):
        return np.array([0, 0])

    @staticmethod
    def friction_bound(u_nu):
        return 0


def main(config: Config):
    """
    Entrypoint to example.

    To see result of simulation you need to call from python `main(Config().init())`.
    """

    setup = DynamicSetup(mesh_type="cross")
    runner = TimeDependentProblemSolver(setup, solving_method="schur")
    n_steps = 32 if not config.test else 10

    states = runner.solve(
        n_steps=n_steps,
        output_step=(0, n_steps),
        verbose=True,
        initial_displacement=setup.initial_displacement,
        initial_velocity=setup.initial_velocity,
    )
    for state in states:
        Drawer(state=state, config=config).draw(show=config.show, save=config.save)


if __name__ == "__main__":
    main(Config().init())
