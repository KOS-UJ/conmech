"""
Created at 18.02.2021
"""
import math
from typing import Optional

import numpy as np
import scipy.optimize

from conmech.dynamics.statement import (
    Statement,
    TemperatureStatement,
    PiezoelectricStatement,
    StaticPoissonStatement,
)
from conmech.scenarios.problems import ContactLaw
from conmech.solvers.solver import Solver
from conmech.solvers.solver_methods import make_cost_functional


class Optimization(Solver):
    def __init__(
        self,
        statement: Statement,
        body: "Body",
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
        if statement.dimension >= 2:  # TODO
            self.loss = make_cost_functional(
                normal_condition=contact_law.potential_normal_direction,
                tangential_condition=contact_law.potential_tangential_direction
                if hasattr(contact_law, "potential_tangential_direction")
                else None,
                tangential_condition_bound=friction_bound,
                variable_dimension=statement.dimension,
                problem_dimension=body.mesh.dimension,
            )
        elif isinstance(statement, TemperatureStatement):
            self.loss = make_cost_functional(
                tangential_condition=contact_law.h_temp,
                normal_condition=contact_law.temp_exchange,
                normal_condition_bound=-1,
            )
        elif isinstance(statement, PiezoelectricStatement):
            self.loss = make_cost_functional(
                tangential_condition=contact_law.electric_charge_tangetial,
                tangential_condition_bound=-1,
                normal_condition=None,
                variable_dimension=statement.dimension,
                problem_dimension=body.mesh.dimension,
            )
        elif isinstance(statement, StaticPoissonStatement):
            self.loss = make_cost_functional(
                normal_condition=contact_law.potential_normal_direction,
                variable_dimension=statement.dimension,
                problem_dimension=body.mesh.dimension,
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
        velocity: np.ndarray,
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
            self.body.mesh.nodes,
            self.body.mesh.contact_boundary,
            self.body.mesh.boundaries.contact_normals,
            self.lhs,
            self.rhs,
            displacement,
            self.time_step,
        )

        while norm >= fixed_point_abs_tol:
            if method.lower() in (  # TODO
                "quasi secant method",
                "limited memory quasi secant method",
                "quasi secant method limited memory",
                "qsm",
                "qsmlm",
            ):
                # pylint: disable=import-outside-toplevel,import-error)
                from kosopt import qsmlm

                solution = qsmlm.minimize(self.loss, solution, args=args, maxiter=maxiter)
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
            norm = np.linalg.norm(np.subtract(solution, old_solution))
            old_solution = solution.copy()
        return solution
