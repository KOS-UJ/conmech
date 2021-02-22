"""
Created at 18.02.2021
"""

import scipy.optimize
import numpy as np
from simulation.solvers.solver_methods import make_f
from simulation.solvers.solver import Solver


class Direct(Solver):

    def __init__(self, grid, inner_forces, outer_forces, mu_coef, lambda_coef, contact_law, friction_bound):
        super().__init__(grid, inner_forces, outer_forces, mu_coef, lambda_coef, contact_law, friction_bound)

        self.f = make_f(jnZ=contact_law.subderivative_normal_direction,
                        jtZ=contact_law.regularized_subderivative_tangential_direction,
                        h=friction_bound
                        )

    def solve(self, initial_guess: np.ndarray) -> np.ndarray:
        result = scipy.optimize.fsolve(
            self.f, initial_guess,
            args=(self.grid.independent_num(), self.grid.BorderEdgesC, self.grid.Edges,
                  self.grid.Points, self.B, self.forces.Zero, self.forces.One))
        return np.asarray(result)
