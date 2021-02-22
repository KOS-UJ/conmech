"""
Created at 18.02.2021
"""

import numpy as np
from simulation.solvers.solver_methods import make_f


class Validator:

    def __init__(self, solver):
        self.B = solver.B
        self.forces = solver.forces
        self.f = make_f(jnZ=solver.contact_law.subderivative_normal_direction,
                        jtZ=solver.contact_law.regularized_subderivative_tangential_direction,
                        h=solver.friction_bound
                        )

    def validate(self, state, displacement) -> float:
        quality_inv = np.linalg.norm(
            self.f(
                displacement, state.grid.independent_num, state.grid.BorderEdgesC, state.grid.Edges,
                state.grid.Points, self.B, self.forces.Zero, self.forces.One)
        )
        quality = quality_inv ** -1
        return quality

    def check_quality(self, state, displacement, previous_quality: float = None) -> float:
        quality = self.validate(state, displacement)
        if previous_quality is not None and previous_quality == quality:
            raise RuntimeError("Can't find a solution! ")
        return quality
