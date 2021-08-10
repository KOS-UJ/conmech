"""
Created at 21.08.2019
"""
from dataclasses import dataclass

import numpy as np

from conmech.problem_solver import Static as StaticProblemSolver
from conmech.problems import Static
from examples.p_slope_contact_law import PSlopeContactLaw
from utils.drawer import Drawer


@dataclass()
class StaticSetup(Static):
    grid_height: ... = 1
    cells_number: ... = (2, 5)
    inner_forces: ... = np.array([-0.2, -0.2])
    outer_forces: ... = np.array([0, 0])
    mu_coef: ... = 4
    lambda_coef: ... = 4
    contact_law: ... = PSlopeContactLaw

    @staticmethod
    def friction_bound(u_nu):
        return 0


if __name__ == '__main__':
    setup = StaticSetup()
    runner = StaticProblemSolver(setup, 'direct')

    state = runner.solve(verbose=True)
    Drawer(state).draw()
