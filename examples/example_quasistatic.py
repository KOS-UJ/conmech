"""
Created at 21.08.2019
"""
#%%
from dataclasses import dataclass

import numpy as np

from conmech.problem_solver import Quasistatic as QuasistaticProblemSolver
from conmech.problems import Quasistatic
from examples.p_slope_contact_law import make_slope_contact_law
from conmech.utils.drawer import Drawer


@dataclass()
class QuasistaticSetup(Quasistatic):
    grid_height: ... = 1
    cells_number: ... = (2, 5)
    inner_forces: ... = np.array([-0.2, -0.2])
    outer_forces: ... = np.array([0, 0])
    mu_coef: ... = 4
    lambda_coef: ... = 4
    th_coef: ... = 4
    ze_coef: ... = 4
    time_step: ... = 0.1
    contact_law: ... = make_slope_contact_law(slope=1e1)

    @staticmethod
    def friction_bound(u_nu):
        return 0


if __name__ == '__main__':
    setup = QuasistaticSetup()
    runner = QuasistaticProblemSolver(setup, solving_method='schur')

    states = runner.solve(n_steps=32, output_step=(0, 32), verbose=True)
    for state in states:
        Drawer(state).draw()

# %%
