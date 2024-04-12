"""
Created at 18.02.2021
"""

from typing import Optional, Callable

import numpy as np
import scipy.optimize

from conmech.dynamics.statement import Statement
from conmech.scenarios.problems import ContactLaw
from conmech.scene.body_forces import BodyForces
from conmech.solvers._solvers import SolversRegistry
from conmech.solvers.solver import Solver
from conmech.solvers.solver_methods import make_equation


@SolversRegistry.register("dynamic", "direct")
@SolversRegistry.register("static", "direct")
class Direct(Solver):
    def __init__(
        self,
        statement: Statement,
        body: BodyForces,
        time_step: float,
        contact_law: Optional[ContactLaw] = None,
        friction_bound: Optional[Callable[[float], float]] = None,
    ):
        super().__init__(
            statement,
            body,
            time_step,
            contact_law,
            friction_bound,
        )
        self.equation: Optional[Callable] = None

        if contact_law is not None:
            self.equation = make_equation(
                jn=contact_law.subderivative_normal_direction,
                jt=contact_law.regularized_subderivative_tangential_direction,
                h_functional=friction_bound,
            )

    def __str__(self) -> str:
        return "direct"

    @property
    def node_relations(self) -> np.ndarray:
        return self.statement.left_hand_side.data

    @property
    def node_forces(self) -> np.ndarray:
        return self.statement.right_hand_side

    def _solve_impl(self, initial_guess: np.ndarray, **kwargs) -> np.ndarray:
        if self.equation is not None:
            result = scipy.optimize.fsolve(
                self.equation,
                initial_guess,
                args=(
                    self.body.mesh.nodes,
                    self.body.mesh.contact_boundary,
                    self.body.mesh.boundaries.contact_normals,
                    self.node_relations,
                    self.node_forces,
                ),
            )
        else:
            result = np.linalg.solve(self.node_relations, self.node_forces)
        return np.asarray(result)
