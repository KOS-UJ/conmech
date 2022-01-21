"""
Created at 18.02.2021
"""
import numpy as np

from conmech.matrices import Matrices
from conmech.f import F


class Solver:

    def __init__(
            self,
            grid,
            inner_forces,
            outer_forces,
            coefficients,
            time_step,
            contact_law,
            friction_bound
    ):
        self.coefficients = coefficients
        self.contact_law = contact_law
        self.friction_bound = friction_bound

        self.grid = grid

        # Added
        self.time_step = time_step
        self.currentTime = 0
        self.u_vector = np.zeros([self.grid.independent_num * 2])
        self.v_vector = np.zeros([self.grid.independent_num * 2])

        self.B = Matrices.construct_B(grid, coefficients.mu, coefficients.lambda_)

        self.forces = F(grid, inner_forces, outer_forces)
        self.forces.setF()

    def __str__(self):
        raise NotImplementedError()

    def iterate(self, velocity):
        self.v_vector = velocity.reshape(-1)
        self.u_vector = self.u_vector + self.time_step * self.v_vector

    def solve(self, initial_guess, **kwargs):
        raise NotImplementedError()
