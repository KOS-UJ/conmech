"""
Created at 18.02.2021
"""
import math
import numpy as np
import scipy.optimize

from conmech.dynamics.statement import (
    StaticDisplacementStatement,
    TemperatureStatement,
    PiezoelectricStatement,
)
from conmech.solvers.solver import Solver
from conmech.solvers.solver_methods import (
    make_cost_functional,
    make_cost_functional_2023,
    make_cost_functional_temperature,
    make_cost_functional_piezoelectricity,
)


class Optimization(Solver):
    def __init__(
        self,
        statement,
        body,
        time_step,
        contact_law,
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
        elif isinstance(statement, TemperatureStatement):
            self.loss = make_cost_functional_temperature(
                h_functional=contact_law.h_temp,
                hn=contact_law.h_nu,
                ht=contact_law.h_tau,
                r=contact_law.temp_exchange,
            )
        elif isinstance(statement, PiezoelectricStatement):
            self.loss = make_cost_functional_piezoelectricity(
                h_functional=contact_law.h_temp,
                hn=contact_law.h_nu,
                ht=contact_law.h_tau,
            )
        else:
            raise ValueError(f"Unknown statement: {statement}")

    def __str__(self):
        raise NotImplementedError()

    @property
    def node_relations(self) -> np.ndarray:
        raise NotImplementedError()

    @property
    def node_forces(self) -> np.ndarray:
        raise NotImplementedError()

    def _solve(
        self,
        initial_guess: np.ndarray,
        *,
        velocity: np.ndarray,
        displacement: np.ndarray,
        fixed_point_abs_tol: float = math.inf,
        **kwargs,
    ) -> np.ndarray:
        norm = math.inf
        solution = np.squeeze(initial_guess.copy().reshape(1, -1))
        velocity = np.squeeze(velocity.copy().reshape(1, -1))
        displacement = np.squeeze(displacement.copy().reshape(1, -1))
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
                    displacement,
                    self.time_step
                ),
                method="BFGS",
                options={"disp": False, "maxiter": len(initial_guess) * 1e5},
                tol=1e-12,
            )
            solution = result.x

            norm = np.linalg.norm(np.subtract(solution, old_solution))
            old_solution = solution.copy()
        return solution
