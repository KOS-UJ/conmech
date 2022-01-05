"""
Created at 21.08.2019
"""
from dataclasses import dataclass

import numpy as np

from conmech.problem_solver import Dynamic as DynamicProblemSolver
from conmech.problems import Dynamic
from examples.p_slope_contact_law import make_slope_contact_law
from utils.drawer import Drawer


@dataclass()
class DynamicSetup(Dynamic):
    grid_height: ... = 1
    cells_number: ... = (2, 5)
    inner_forces: ... = np.array([0, 0.2])
    outer_forces: ... = np.array([0, 0])
    mu_coef: ... = 4
    lambda_coef: ... = 4
    th_coef: ... = 4
    ze_coef: ... = 4
    time_step: ... = 0.1
    contact_law: ... = make_slope_contact_law(slope=0)

    @staticmethod
    def friction_bound(u_nu):
        return 0


if __name__ == '__main__':
    setup = DynamicSetup()
    runner = DynamicProblemSolver(setup, solving_method='schur')

    states = runner.solve(n_steps=32, output_step=(0, 32), verbose=True)
    displacement_vector = states[-1].displacement.T.reshape(1, -1)[0]
    import numpy
    numpy.set_printoptions(precision=8, suppress=True)
    print(repr(displacement_vector))
    for state in states:
        Drawer(state).draw()
