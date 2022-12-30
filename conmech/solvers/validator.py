"""
Created at 18.02.2021
"""

import numpy as np

from conmech.helpers import jxh
from conmech.solvers.solver_methods import make_equation


class Validator:
    def __init__(self, solver, error_tolerance: float = 1):
        self.error_tolerance = error_tolerance
        self.elasticity = solver.elasticity
        self.rhs = make_equation(
            jn=solver.contact_law.subderivative_normal_direction,
            jt=solver.contact_law.regularized_subderivative_tangential_direction,
            h_functional=solver.friction_bound,
        )

    def validate(self, state, solution) -> float:
        quality_inv = np.linalg.norm(
            self.rhs(
                solution,
                state.body.initial_nodes,
                state.body.contact_boundary,
                jxh.to_dense_np(self.elasticity),
                state.body.get_integrated_forces_vector_np(),
            )
        )
        if quality_inv == 0:
            quality = np.inf
        else:
            quality = quality_inv**-1
        return quality

    def check_quality(self, state, solution, previous_quality: float = None) -> float:
        quality = self.validate(state, solution)
        if previous_quality is not None and previous_quality == quality:
            raise RuntimeError("Can't find a solution! ")
        return quality
