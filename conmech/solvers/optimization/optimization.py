"""
Created at 18.02.2021
"""
import math
from typing import Optional

import numpy as np
import scipy.optimize

from conmech.dynamics.statement import StaticDisplacementStatement, Statement
from conmech.scenarios.problems import ContactLaw
from conmech.scene.body_forces import BodyForces
from conmech.solvers.solver import Solver
from conmech.solvers.solver_methods import make_cost_functional
from conmech.solvers.solver_methods import make_cost_functional_temperature


class Optimization(Solver):
    def __init__(
        self,
        statement: Statement,
        body: BodyForces,
        time_step: float,
        contact_law: Optional[ContactLaw],
        friction_bound,
    ):
        super().__init__(
            statement,
            body,
            time_step,
            contact_law,
            friction_bound,
        )
        if statement.dimension == 2:  # TODO
            self.loss = make_cost_functional(
                jn=contact_law.potential_normal_direction,
                jt=contact_law.potential_tangential_direction
                if hasattr(contact_law, "potential_tangential_direction")
                else None,
                h_functional=friction_bound,
            )
        else:
            self.loss = make_cost_functional_temperature(
                h_functional=contact_law.h_temp,
                hn=contact_law.h_nu,
                ht=contact_law.h_tau,
            )

    def __str__(self):
        raise NotImplementedError()

    @property
    def node_relations(self) -> np.ndarray:
        raise NotImplementedError()

    @property
    def node_forces(self) -> np.ndarray:
        raise NotImplementedError()

    def solve(
        self,
        initial_guess: np.ndarray,
        *,
        velocity: np.ndarray,
        fixed_point_abs_tol: float = math.inf,
        **kwargs,
    ) -> np.ndarray:
        norm = math.inf
        solution = np.squeeze(initial_guess.copy().reshape(1, -1))
        velocity = np.squeeze(velocity.copy().reshape(1, -1))
        old_solution = np.squeeze(initial_guess.copy().reshape(1, -1))

        while norm >= fixed_point_abs_tol:
            result = scipy.optimize.minimize(
                self.loss,
                solution,
                args=(
                    self.body.mesh.initial_nodes,
                    self.body.mesh.contact_boundary,
                    self.node_relations,
                    self.node_forces,
                    old_solution
                    if isinstance(self.statement, StaticDisplacementStatement)
                    else velocity,
                ),
                method="BFGS",
                options={"disp": True, "maxiter": len(initial_guess) * 1e5},
                tol=1e-12,
            )
            solution = result.x

            norm = np.linalg.norm(np.subtract(solution, old_solution))
            old_solution = solution.copy()
        return solution
