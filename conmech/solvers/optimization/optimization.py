# CONMECH @ Jagiellonian University in Kraków
#
# Copyright (C) 2021-2024  Piotr Bartman-Szwarc <piotr.bartman@uj.edu.pl>
#
# This program is free software; you can redistribute it and/or
# modify it under the terms of the GNU General Public License
# as published by the Free Software Foundation; either version 3
# of the License, or (at your option) any later version.
#
# This program is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
# GNU General Public License for more details.
#
# You should have received a copy of the GNU General Public License
# along with this program; if not, write to the Free Software
# Foundation, Inc., 51 Franklin Street, Fifth Floor, Boston, MA  02110-1301,
# USA.
import math
import time
from typing import Optional

import numba
import numpy as np
import scipy.optimize

from conmech.struct.types import *
from conmech.dynamics.statement import (
    Statement,
    StaticPoissonStatement,
    WaveStatement,
)
from conmech.dynamics.contact.contact_law import PotentialOfContactLaw
from conmech.dynamics.contact.interior_contact_law import InteriorContactLaw
from conmech.solvers.solver import Solver
from conmech.solvers.solver_methods import (
    make_cost_functional,
    make_equation,
    make_subgradient, make_subgradient_dc,
)


QSMLM_NAMES = {
    "quasi secant method",
    "limited memory quasi secant method",
    "quasi secant method limited memory",
    "qsm",
    "qsmlm",
}
QSMLM_NAMES = {"dc " + name for name in QSMLM_NAMES}.union(QSMLM_NAMES)
GLOBAL_QSMLM_NAMES = {
    "global quasi secant method",
    "global limited memory quasi secant method",
    "global quasi secant method limited memory",
    "globqsm",
    "globqsmlm",
}
GLOBAL_QSMLM_NAMES = {"dc " + name for name in GLOBAL_QSMLM_NAMES}.union(GLOBAL_QSMLM_NAMES)


class Optimization(Solver):
    def __init__(
        self,
        statement: Statement,
        body: "Body",
        time_step: float,
        contact_law: Optional[PotentialOfContactLaw],
        driving_vector,
    ):
        super().__init__(
            statement,
            body,
            time_step,
            contact_law,
            driving_vector,
        )
        self.loss = make_cost_functional(
            normal_condition=contact_law.potential_normal_direction,
            normal_condition_bound=contact_law.normal_bound,
            tangential_condition=contact_law.potential_tangential_direction,
            tangential_condition_bound=contact_law.tangential_bound,
            variable_dimension=statement.dimension_out,
            problem_dimension=statement.dimension_in,
        )
        if hasattr(contact_law, "subderivative_normal_direction"):  # TODO
            self.subgradient = make_subgradient(
                normal_condition=contact_law.subderivative_normal_direction,
            )
        else:
            self.subgradient = None
        if hasattr(contact_law, "sub2derivative_normal_direction"):  # TODO
            self.sub2gradient = make_subgradient_dc(
                normal_condition=contact_law.subderivative_normal_direction,
                normal_condition_sub2=contact_law.sub2derivative_normal_direction,
                # only_boundary=True,
            )
        else:
            self.sub2gradient = None
        if isinstance(statement, WaveStatement):
            if isinstance(contact_law, InteriorContactLaw):
                self.loss = make_equation(  # TODO!
                    jn=None,
                    contact=contact_law.potential_normal_direction,
                )

        if isinstance(statement, StaticPoissonStatement):
            self.loss = make_cost_functional(
                normal_condition=(
                    contact_law.potential_normal_direction if contact_law is not None else None
                ),
                variable_dimension=statement.dimension_out,
                problem_dimension=statement.dimension_in,
            )
        self.minimizer = None

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
        start = time.time()
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
            self.rhs if len(self.rhs.shape) == 1 else self.rhs[0],  # TODO
            displacement,
            np.ascontiguousarray(self.body.dynamics.acceleration_operator.SM1.data),
            self.time_step,
        )

        loss = []
        sols = []
        sols.append(solution)
        loss.append(self.loss(solution, *args)[0])
        self.computation_time += time.time() - start

        if self.minimizer is None and method.lower() in QSMLM_NAMES.union(GLOBAL_QSMLM_NAMES):
            # pylint: disable=import-outside-toplevel,import-error)
            from kosopt.qsmlm import make_minimizer

            self.minimizer = make_minimizer(
                self.loss,
                self.subgradient,
                self.sub2gradient if method.lower().startswith("dc") else None
            )

        while norm >= fixed_point_abs_tol:
            if method.lower() in QSMLM_NAMES:
                start = time.time()
                solution = self.minimizer(solution, args, 0, 1, maxiter)
                self.computation_time += time.time() - start
                sols.append(solution.copy())
                loss.append(self.loss(solution, *args)[0])
            elif method.lower() in GLOBAL_QSMLM_NAMES:
                # pylint: disable=import-outside-toplevel,import-error)
                from kosopt import subgradient

                solution, comp_time = subgradient.minimize(
                    self.minimizer,
                    self.loss,
                    solution,
                    args,
                    maxiter=maxiter,
                    subgradient=self.subgradient,
                    sub2gradient=self.sub2gradient,
                )
                sols.append(solution.copy())
                loss.append(self.loss(solution, *args)[0])
                self.computation_time += comp_time
            elif method.lower() in (  # TODO
                "discontinuous gradient",
                "discontinuous gradient method",
                "dg",
            ):
                # pylint: disable=import-outside-toplevel,import-error)
                from kosopt import qsmlmi

                solution = qsmlmi.minimize(self.loss, solution, args, 0, 1, 10000)
                sols.append(solution.copy())
                loss.append(self.loss(solution, *args)[0])
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
            else:
                subgrad = None
                if method.startswith("gradiented "):
                    subgrad = self.subgradient
                    method = method[len("gradiented "):]
                start = time.time()
                result = scipy.optimize.minimize(
                    self.loss,
                    solution,
                    args=args,
                    method=method,
                    jac=subgrad,
                    options={"disp": disp, "maxiter": maxiter},
                    tol=tol,
                )
                self.computation_time += time.time() - start
                solution = result.x
                sols.append(solution.copy())
                loss.append(self.loss(solution, *args)[0])

            norm = np.linalg.norm(np.subtract(solution, old_solution))
            old_solution = solution.copy()
            break
        min_index = loss.index(np.min(loss))
        solution = sols[min_index]

        return solution
