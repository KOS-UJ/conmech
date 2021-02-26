"""
Created at 22.02.2021
"""

import numpy as np
from conmech.solvers.optimization.optimization import Optimization


class Global(Optimization):

    def __init__(self, grid, inner_forces, outer_forces, mu_coef, lambda_coef, contact_law, friction_bound):
        super().__init__(grid, inner_forces, outer_forces, mu_coef, lambda_coef, contact_law, friction_bound)
        self.__point_relations = np.bmat([[self.B[0, 0], self.B[0, 1]], [self.B[1, 0], self.B[1, 1]]])
        self.__point_forces = np.append(self.forces.Zero, self.forces.One)

    @property
    def point_relations(self) -> np.ndarray:
        return self.__point_relations

    @property
    def point_forces(self) -> np.ndarray:
        return self.__point_forces
