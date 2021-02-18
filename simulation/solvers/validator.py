"""
Created at 18.02.2021
"""

import numpy as np
from simulation.solver import make_f


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
                displacement, state.grid.indNumber(), state.grid.BorderEdgesC, state.grid.Edges,
                state.grid.Points, self.B, self.forces.Zero, self.forces.One)
        )
        quality = quality_inv ** -1
        return quality
