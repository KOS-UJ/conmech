"""
Created at 21.08.2019
"""
from dataclasses import dataclass

import numpy as np

from conmech.helpers.config import Config
from conmech.plotting.drawer import Drawer
from conmech.scenarios.problems import Quasistatic
from conmech.simulations.problem_solver import Quasistatic as QuasistaticProblemSolver
from examples.p_slope_contact_law import make_slope_contact_law


@dataclass()
class QuasistaticSetup(Quasistatic):
    grid_height: ... = 1.0
    elements_number: ... = (2, 5)
    mu_coef: ... = 4
    la_coef: ... = 4
    th_coef: ... = 4
    ze_coef: ... = 4
    time_step: ... = 0.1
    contact_law: ... = make_slope_contact_law(slope=1e1)

    @staticmethod
    def inner_forces(x):
        return np.array([-0.2, -0.2])

    @staticmethod
    def outer_forces(x):
        return np.array([0, 0])

    @staticmethod
    def friction_bound(u_nu):
        return 0

    @staticmethod
    def is_contact(x):
        return x[1] == 0

    @staticmethod
    def is_dirichlet(x):
        return x[0] == 0


def main(show: bool = True, save: bool = False):
    setup = QuasistaticSetup()
    runner = QuasistaticProblemSolver(setup, solving_method="schur")

    states = runner.solve(
        n_steps=32,
        output_step=(0, 32),
        verbose=True,
        initial_displacement=setup.initial_displacement,
        initial_velocity=setup.initial_velocity,
    )
    config = Config()
    for state in states:
        Drawer(state=state, config=config).draw(show=show, save=save)


if __name__ == "__main__":
    main(show=True)
