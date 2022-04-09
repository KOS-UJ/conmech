"""
Created at 18.02.2021
"""

import numpy as np
import scipy.optimize

from conmech.dynamics.statement import StaticStatement
from conmech.solvers._solvers import Solvers
from conmech.solvers.solver import Solver
from conmech.solvers.solver_methods import make_equation


@Solvers.register("static", "direct")
class Direct(Solver):
    def __init__(
        self,
        mesh,
        inner_forces,
        outer_forces,
        body_prop,
        time_step,
        contact_law,
        friction_bound,
    ):
        super().__init__(
            mesh,
            StaticStatement(mesh),
            inner_forces,
            outer_forces,
            body_prop,
            time_step,
            contact_law,
            friction_bound,
        )

        self.equation = make_equation(
            jn=contact_law.subderivative_normal_direction,
            jt=contact_law.regularized_subderivative_tangential_direction,
            h_functional=friction_bound,
        )

    def __str__(self):
        return "direct"

    @property
    def point_relations(self) -> np.ndarray:
        return self.statement.left_hand_side

    @property
    def point_forces(self) -> np.ndarray:
        return self.statement.right_hand_side

    def solve(self, initial_guess: np.ndarray, **kwargs) -> np.ndarray:
        result = scipy.optimize.fsolve(
            self.equation,
            initial_guess,
            args=(
                self.mesh.initial_nodes,
                self.mesh.contact_boundary,
                self.point_relations,
                self.point_forces,
            ),
        )
        return np.asarray(result)
