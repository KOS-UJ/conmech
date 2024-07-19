"""
Created at 18.02.2021
"""

import math
from typing import Optional

import numba
import numpy as np
import scipy.optimize

from conmech.dynamics.statement import (
    Statement,
    TemperatureStatement,
    PiezoelectricStatement,
    StaticPoissonStatement, WaveStatement,
)
from conmech.dynamics.contact.contact_law import ContactLaw, InteriorContactLaw, \
    PotentialOfContactLaw
from conmech.solvers.solver import Solver
from conmech.solvers.solver_methods import make_cost_functional, make_equation, \
    make_cost_functional_2


class Optimization(Solver):
    def __init__(
        self,
        statement: Statement,
        body: "Body",
        time_step: float,
        contact_law: Optional[PotentialOfContactLaw],
        friction_bound,
        driving_vector,
    ):
        super().__init__(
            statement,
            body,
            time_step,
            contact_law,
            friction_bound,
            driving_vector,
        )
        if statement.dimension_in > 1 and statement.dimension_out > 1:
            self.loss = make_cost_functional(
                normal_condition=contact_law.potential_normal_direction,
                tangential_condition=contact_law.potential_tangential_direction,
                tangential_condition_bound=contact_law.tangential_bound,
                variable_dimension=statement.dimension_out,
                problem_dimension=statement.dimension_in
            )
        elif isinstance(statement, TemperatureStatement):
            self.loss = make_cost_functional(
                tangential_condition=contact_law.potential_tangential_direction,
                normal_condition=contact_law.potential_normal_direction,
                normal_condition_bound=contact_law.normal_bound,
            )
        elif isinstance(statement, PiezoelectricStatement):
            self.loss = make_cost_functional(
                tangential_condition=contact_law.potential_tangential_direction,
                tangential_condition_bound=contact_law.tangential_bound,
                normal_condition=contact_law.potential_normal_direction,
                variable_dimension=statement.dimension_out,
                problem_dimension=statement.dimension_in
            )
        elif isinstance(statement, StaticPoissonStatement):
            self.loss = make_cost_functional(
                normal_condition=contact_law.potential_normal_direction if contact_law is not None else None,
                variable_dimension=statement.dimension_out,
                problem_dimension=statement.dimension_in,
            )
        elif isinstance(statement, WaveStatement):
            if isinstance(contact_law, InteriorContactLaw):
                self.loss = make_equation(  # TODO!
                    jn=None,
                    contact=contact_law.potential_normal_direction,
                    h_functional=friction_bound,
                )
            else:
                self.loss = make_cost_functional_2(
                    normal_condition=contact_law.potential_normal_direction,
                    variable_dimension=statement.dimension_out,
                    problem_dimension=statement.dimension_in,
                )
        else:
            raise ValueError(f"Unknown statement: {statement}")

    def __str__(self):
        raise NotImplementedError()

    @property
    def lhs(self) -> np.ndarray:
        raise NotImplementedError()

    @property
    def rhs(self) -> np.ndarray:
        raise NotImplementedError()

    def _solve_impl(
        self,
        initial_guess: np.ndarray,
        *,
        variable_old: np.ndarray,
        displacement: np.ndarray,
        method="BFGS",
        fixed_point_abs_tol: float = math.inf,
        **kwargs,
    ) -> np.ndarray:
        norm = math.inf
        solution = np.squeeze(initial_guess.copy().reshape(1, -1))
        displacement = np.squeeze(displacement.copy().reshape(1, -1))
        old_solution = np.squeeze(initial_guess.copy().reshape(1, -1))
        disp = kwargs.get("disp", False)
        maxiter = kwargs.get("maxiter", int(len(initial_guess) * 1e9))
        tol = kwargs.get("tol", 1e-12)
        args = (
            variable_old,
            self.body.mesh.nodes,
            self.body.mesh.contact_boundary,
            self.body.mesh.boundaries.contact_normals,
            self.lhs,
            self.rhs,
            displacement,
            np.ascontiguousarray(self.body.dynamics.acceleration_operator.SM1.data),
            self.time_step,
        )

        loss = []
        sols = []
        sols.append(solution)
        loss.append(self.loss(solution, *args)[0])

        while norm >= fixed_point_abs_tol:
            if method.lower() in (
                "quasi secant method",
                "limited memory quasi secant method",
                "quasi secant method limited memory",
                "qsm",
                "qsmlm",
            ):
                # pylint: disable=import-outside-toplevel,import-error)
                from kosopt import qsmlm

                solution = qsmlm.minimize(self.loss, solution, args=args, maxiter=maxiter)
                sols.append(solution.copy())
            elif method.lower() == "constrained":
                contact_nodes_count = self.body.mesh.boundaries.contact_nodes_count

                @numba.njit()
                def constr(x):
                    offset = len(x) // 2
                    t = x[offset : offset + contact_nodes_count]
                    return np.min(t)

                maxiter = kwargs.get("maxiter", int(len(initial_guess) * 1e9))
                tol = kwargs.get("tol", 1e-12)
                result = scipy.optimize.minimize(
                    self.loss,
                    solution,
                    args=args,
                    options={"disp": disp, "maxiter": maxiter},
                    tol=tol,
                    constraints=({"type": "ineq", "fun": constr}),
                )
                solution = result.x
                sols.append(solution)
                loss.append(self.loss(solution, *args)[0])
                break
            else:
                result = scipy.optimize.minimize(
                    self.loss,
                    solution,
                    args=args,
                    method=method,
                    options={"disp": disp, "maxiter": maxiter},
                    tol=tol,
                )
                solution = result.x
                sols.append(solution.copy())
                loss.append(self.loss(solution, *args)[0])
                break

            norm = np.linalg.norm(np.subtract(solution, old_solution))
            old_solution = solution.copy()
        min_index = loss.index(np.min(loss))
        solution = sols[min_index]

        return solution
