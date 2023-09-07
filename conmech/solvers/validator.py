"""
Created at 18.02.2021
"""

import numpy as np

from conmech.solvers.solver import Solver
from conmech.solvers.solver_methods import make_equation
from conmech.state.state import State


class Validator:
    def __init__(self, solver: Solver, error_tolerance: float = 1):
        self.error_tolerance: float = error_tolerance
        self.elasticity: np.ndarray = solver.elasticity
        self.rhs: callable
        if solver.contact_law is None:
            self.rhs = make_equation(None, None, None)
        else:
            self.rhs = make_equation(
                jn=solver.contact_law.subderivative_normal_direction,
                jt=solver.contact_law.regularized_subderivative_tangential_direction,
                h_functional=solver.friction_bound,
            )

    def validate(self, state: State, solution: np.ndarray) -> float:
        quality_inv = np.linalg.norm(
            self.rhs(
                solution,
                state.body.mesh.initial_nodes,
                state.body.mesh.contact_boundary,
                self.elasticity,
                state.body.dynamics.force.integrate(time=state.time),
            )
        )
        if quality_inv == 0:
            quality = np.inf
        else:
            quality = quality_inv**-1
        return quality

    def check_quality(
        self, state: State, solution: np.ndarray, previous_quality: float = None
    ) -> float:
        quality = self.validate(state, solution)
        if previous_quality is not None and previous_quality == quality:
            raise RuntimeError("Can't find a solution! ")
        return quality
