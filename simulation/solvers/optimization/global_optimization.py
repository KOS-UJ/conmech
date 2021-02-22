"""
Created at 22.02.2021
"""

import numpy as np
from simulation.solvers.optimization.optimization import Optimization


class Global(Optimization):

    def __init__(self, grid, inner_forces, outer_forces, mu_coef, lambda_coef, contact_law, friction_bound):
        super().__init__(grid, inner_forces, outer_forces, mu_coef, lambda_coef, contact_law, friction_bound)
        self._C = np.bmat([[self.B[0, 0], self.B[0, 1]], [self.B[1, 0], self.B[1, 1]]])
        self._E = np.append(self.forces.Zero, self.forces.One)

    @property
    def C(self):
        return self._C

    @property
    def E(self):
        return self._E
