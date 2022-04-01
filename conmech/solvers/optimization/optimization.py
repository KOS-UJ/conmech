"""
Created at 18.02.2021
"""
import math
import numpy as np
import scipy.optimize

from conmech.solvers.solver import Solver
from conmech.solvers.solver_methods import make_cost_functional
from conmech.solvers.solver_methods import make_cost_functional_temperature


class Optimization(Solver):
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
            inner_forces,
            outer_forces,
            body_prop,
            time_step,
            contact_law,
            friction_bound,
        )
        self.loss = make_cost_functional(
            jn=contact_law.potential_normal_direction,
            jt=contact_law.potential_tangential_direction
            if hasattr(contact_law, "potential_tangential_direction")
            else None,
            h=friction_bound,
        )
        if hasattr(contact_law, "h_temp"):
            self.loss_temp = make_cost_functional_temperature(
                h=contact_law.h_temp, hn=contact_law.h_nu, ht=contact_law.h_tau
            )

    def __str__(self):
        raise NotImplementedError()

    @property
    def point_relations(self) -> np.ndarray:
        raise NotImplementedError()

    @property
    def point_forces(self) -> np.ndarray:
        raise NotImplementedError()

    def solve(
            self,
            initial_guess: np.ndarray,
            *,
            fixed_point_abs_tol: float = math.inf,
            **kwargs
    ) -> np.ndarray:
        norm = math.inf
        solution = np.squeeze(initial_guess.copy().reshape(1, -1))
        old_solution = np.squeeze(initial_guess.copy().reshape(1, -1))
        while norm >= fixed_point_abs_tol:
            result = scipy.optimize.minimize(
                self.loss,
                solution,
                args=(
                    old_solution,
                    self.mesh.initial_nodes,
                    self.mesh.boundaries.contact,
                    self.point_relations,
                    self.point_forces,
                ),
                method="BFGS",
                options={"disp": True, "maxiter": len(initial_guess) * 1e5},
                tol=1e-12,
            )
            solution = result.x
            norm = np.linalg.norm(np.subtract(solution, old_solution))
            old_solution = solution.copy()
        return solution

    def solve_t(self, initial_guess: np.ndarray, velocity: np.ndarray) -> np.ndarray:
        loss_args = (
            self.mesh.initial_nodes,
            self.mesh.boundaries.contact,
            self.T,  # TODO #48
            self.Q,  # TODO #48
            velocity,
        )
        # TODO #33
        result = scipy.optimize.minimize(
            self.loss_temp,
            initial_guess,
            args=loss_args,
            method="BFGS",
            options={"disp": True, "maxiter": len(initial_guess) * 1e5},
            tol=1e-12,
        )
        result = result.x
        return result
