"""
Created at 18.02.2021
"""

import scipy.optimize
import numpy as np
from conmech.solvers.solver import Solver
from conmech.solvers.solver_methods import make_L2


class Optimization(Solver):

    def __init__(self, grid, inner_forces, outer_forces, mu_coef,
                 lambda_coef, th_coef, ze_coef, time_step, contact_law, friction_bound):
        super().__init__(grid, inner_forces, outer_forces, mu_coef,
                         lambda_coef, th_coef, ze_coef, time_step, contact_law, friction_bound)
        self.loss = make_L2(jn=contact_law.potential_normal_direction)

    def __str__(self):
        raise NotImplementedError()

    @property
    def point_relations(self) -> np.ndarray:
        raise NotImplementedError()

    @property
    def point_forces(self) -> np.ndarray:
        raise NotImplementedError()

    def solve(self, initial_guess: np.ndarray) -> np.ndarray:
        result = scipy.optimize.minimize(
            self.loss,
            initial_guess,
            args=(
                self.grid.independent_num,
                self.grid.BorderEdgesC,
                self.grid.Edges,
                self.grid.Points,
                self.point_relations,
                self.point_forces
            ),
            method='BFGS',
            options={'disp': True, 'maxiter': len(initial_guess) * 1e5},
            tol=1e-12
        )
        result = result.x
        return result
