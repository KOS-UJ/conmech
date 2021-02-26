"""
Created at 18.02.2021
"""

from conmech.matrices import Matrices
from conmech.f import F


class Solver:

    def __init__(self, grid, inner_forces, outer_forces, mu_coef, lambda_coef, contact_law, friction_bound):
        self.mu_coef = mu_coef
        self.lambda_coef = lambda_coef
        self.contact_law = contact_law
        self.friction_bound = friction_bound

        self.grid = grid
        # TODO
        # self.time_step = time_step
        # self.currentTime = 0

        self.B = Matrices.construct_B(grid, mu_coef, lambda_coef)
        self.forces = F(grid, inner_forces, outer_forces)
        self.forces.setF()

    def solve(self, initial_guess):
        raise NotImplementedError()
