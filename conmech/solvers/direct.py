"""
Created at 18.02.2021
"""

import numpy as np
import scipy.optimize

from conmech.solvers._solvers import Solvers
from conmech.solvers.solver import Solver
from conmech.solvers.solver_methods import make_equation


@Solvers.register("static", "direct")
class Direct(Solver):
    def __init__(
            self,
            grid,
            inner_forces,
            outer_forces,
            body_prop,
            time_step,
            contact_law,
            friction_bound,
    ):
        super().__init__(
            grid,
            inner_forces,
            outer_forces,
            body_prop,
            time_step,
            contact_law,
            friction_bound,
        )

        self.f = make_equation(
            jn=contact_law.subderivative_normal_direction,
            jt=contact_law.regularized_subderivative_tangential_direction,
            h=friction_bound,
        )

    def __str__(self):
        return "direct"

    def solve(self, initial_guess: np.ndarray, **kwargs) -> np.ndarray:
        result = scipy.optimize.fsolve(
            self.f,
            initial_guess,
            args=(
                self.mesh.initial_nodes,
                self.mesh.contact_boundary,
                self.elasticity,
                self.forces.forces_vector,
            ),
        )
        return np.asarray(result)
