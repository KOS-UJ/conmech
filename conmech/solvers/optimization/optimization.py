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
from conmech.scenarios.problems import ContactLaw, InteriorContactLaw
from conmech.solvers.solver import Solver
from conmech.solvers.solver_methods import make_cost_functional, make_equation, \
    make_cost_functional_2


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
                tangential_condition=(
                    contact_law.potential_tangential_direction
                    if hasattr(contact_law, "potential_tangential_direction")
                    else None
                ),
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
        elif isinstance(statement, WaveStatement):
            if isinstance(contact_law, InteriorContactLaw):
                self.loss = make_equation(  # TODO!
                    jn=contact_law.subderivative_normal_direction,
                    jt=contact_law.regularized_subderivative_tangential_direction,
                    contact=contact_law.general_contact_condition,
                    h_functional=friction_bound,
                )
            else:
                self.loss = make_cost_functional_2(
                    contact=contact_law.general_contact_condition,
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
            np.ascontiguousarray(self.body.dynamics.acceleration_operator.SM1.data),
            self.time_step,
        )

        loss = []
        sols = []
        sols.append(solution)
        loss.append(self.loss(solution, *args)[0])

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
                sols.append(solution.copy())
                loss.append(self.loss(solution, *args)[0])

                ### TODO
                ind = self.lhs.shape[0]
                response = np.zeros(ind)
                for i in range(ind):
                    response[i] = self.contact_law.general_contact_condition(
                        displacement[i] + solution[i] * self.time_step,
                        solution[i])
                validation = np.dot(self.lhs, solution[:ind]) \
                             + np.dot(self.body.dynamics.volume_at_nodes,
                                      response) \
                             - self.rhs
                valid = np.linalg.norm(validation)
                if valid < 0.1:
                    print("Validation:", valid)
                    break
                else:
                    print("Trying again:", valid)

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

                ### TODO
                # ind = self.lhs.shape[0]
                # response = np.zeros(ind)
                # for i in range(ind):
                #     response[i] = self.contact_law.general_contact_condition(
                #         displacement[i] + solution[i] * self.time_step,
                #         solution[i])
                # validation = np.dot(self.lhs, solution[:ind]) \
                #              + np.dot(self.body.dynamics.volume_at_nodes,
                #                       response) \
                #              - self.rhs
                # valid = np.linalg.norm(validation)
                # if valid < 0.1:
                #     print("Validation:", valid)
                #     break
                # else:
                #     print("Trying again:", valid)

            norm = np.linalg.norm(np.subtract(solution, old_solution))
            old_solution = solution.copy()
        min_index = loss.index(np.min(loss))
        solution = sols[min_index]
        return solution
