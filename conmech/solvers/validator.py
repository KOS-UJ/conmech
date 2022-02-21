"""
Created at 18.02.2021
"""

import numpy as np
from conmech.solvers.solver_methods import make_f


class Validator:
    def __init__(self, solver, error_tolerance: float = 1):
        self.error_tolerance = error_tolerance
        self.B = solver.B
        self.forces = solver.forces
        self.f = make_f(
            jnZ=solver.contact_law.subderivative_normal_direction,
            jtZ=solver.contact_law.regularized_subderivative_tangential_direction,
            h=solver.friction_bound,
        )

    def validate(self, state, solution) -> float:
        quality_inv = np.linalg.norm(
            self.f(
                solution,
                state.mesh.initial_points,
                state.mesh.boundaries.contact,
                self.B,
                self.forces.Zero[: state.mesh.independent_nodes_conunt],
                self.forces.One[: state.mesh.independent_nodes_conunt],
            )
        )
        quality = quality_inv ** -1
        return quality

    def check_quality(self, state, solution, previous_quality: float = None) -> float:
        quality = self.validate(state, solution)
        if previous_quality is not None and previous_quality == quality:
            raise RuntimeError("Can't find a solution! ")
        return quality
