"""
Created at 21.08.2019
"""
#%%
from dataclasses import dataclass

import numpy as np

from conmech.problem_solver import Static as StaticProblemSolver
from conmech.problems import Static
from examples.p_slope_contact_law import make_slope_contact_law
from conmech.utils.drawer import Drawer


@dataclass()
class StaticSetup(Static):
    grid_height: ... = 1.0
    cells_number: ... = (2, 5)
    inner_forces: ... = np.array([-0.2, -0.2])
    outer_forces: ... = np.array([0, 0])
    mu_coef: ... = 4
    la_coef: ... = 4
    contact_law: ... = make_slope_contact_law(slope=1)

    @staticmethod
    def friction_bound(u_nu):
        return 0

    @staticmethod
    def is_contact(x, y):
        return y == 0

    @staticmethod
    def is_dirichlet(x, y):
        return x == 0


if __name__ == "__main__":
    setup = StaticSetup()
    runner = StaticProblemSolver(setup, "direct")

    state = runner.solve(verbose=True)
    Drawer(state).draw()

# %%
