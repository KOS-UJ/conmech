"""
Created at 21.08.2019
"""
from dataclasses import dataclass

import numpy as np

from conmech.problem_solver import Static as StaticProblemSolver
from conmech.problems import Static
from conmech.utils.drawer import Drawer
from examples.p_slope_contact_law import make_slope_contact_law


@dataclass()
class StaticSetup(Static):
    grid_height: ... = 1.0
    elements_number: ... = (2, 5)
    mu_coef: ... = 4
    la_coef: ... = 4
    contact_law: ... = make_slope_contact_law(slope=1)

    @staticmethod
    def inner_forces(x, y):
        return np.array([-0.2, -0.2])

    @staticmethod
    def outer_forces(x, y):
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


def main(show: bool):
    setup = StaticSetup()
    runner = StaticProblemSolver(setup, "direct")

    state = runner.solve(verbose=True,
                         initial_displacement=setup.initial_displacement)
    Drawer(state).draw(show=show)


if __name__ == "__main__":
    main(show=True)
