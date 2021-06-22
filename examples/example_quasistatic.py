"""
Created at 21.08.2019
"""
from dataclasses import dataclass

import numpy as np

from conmech.problem_solver import Quasistatic as QuasistaticProblemSolver
from conmech.problems import Quasistatic
from examples.p_slope_contact_law import PSlopeContactLaw
from utils.drawer import Drawer


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
    contact_law: ... = PSlopeContactLaw

    @staticmethod
    def friction_bound(u_nu):
        return 0


if __name__ == '__main__':
    setup = QuasistaticSetup()
    runner = QuasistaticProblemSolver(setup, solving_method='schur')

    states = runner.solve(n_steps=100, output_step=(0, 10, 20, 30, 40, 50, 60, 70, 80, 90), verbose=True)
    for state in states:
        Drawer(state).draw()
