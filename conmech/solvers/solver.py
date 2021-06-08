"""
Created at 18.02.2021
"""
import numpy as np

from conmech.matrices import Matrices
from conmech.f import F


class Solver:

    def __init__(self, grid, inner_forces, outer_forces, mu_coef, lambda_coef,
                 th_coef, ze_coef, time_step, contact_law, friction_bound):
        self.mu_coef = mu_coef
        self.lambda_coef = lambda_coef
        self.contact_law = contact_law
        self.friction_bound = friction_bound

        self.grid = grid

        # Added
        self.th_coef = th_coef
        self.ze_coef = ze_coef
        self.time_step = time_step
        self.currentTime = 0
        self.u_vector = np.zeros([self.grid.independent_num * 2])
        self.v_vector = np.zeros([self.grid.independent_num * 2])

        self.A = Matrices.construct_B(grid, th_coef, ze_coef)  # TODO: optionally
        self.B = Matrices.construct_B(grid, mu_coef, lambda_coef)
        self.U = Matrices.construct_U(grid)  # TODO: optionally

        self.forces = F(grid, inner_forces, outer_forces)
        self.forces.setF()

    def __repr__(self):
        raise NotImplementedError()

    def iterate(self, velocity):
        self.v_vector = velocity.reshape(-1)
        self.u_vector = self.u_vector + self.time_step * self.v_vector

    def solve(self, initial_guess):
        raise NotImplementedError()
