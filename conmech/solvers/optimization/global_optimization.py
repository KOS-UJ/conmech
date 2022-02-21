"""
Created 22.02.2021
"""

import numpy as np

from conmech.solvers.optimization.optimization import Optimization
from conmech.solvers._solvers import Solvers


@Solvers.register("*", "global", "global optimization")
class Global(Optimization):

    def __init__(self, grid, inner_forces, outer_forces, coefficients, time_step, contact_law, friction_bound):
        super().__init__(grid, inner_forces, outer_forces, coefficients, time_step, contact_law, friction_bound)
        self.__point_relations = np.bmat([[self.B[0, 0], self.B[0, 1]], [self.B[1, 0], self.B[1, 1]]])
        self.__point_forces = np.append(self.forces.Zero, self.forces.One)

    def __str__(self):
        return "global optimization"

    @property
    def point_relations(self) -> np.ndarray:
        return self.__point_relations

    @property
    def point_forces(self) -> np.ndarray:
        return self.__point_forces
