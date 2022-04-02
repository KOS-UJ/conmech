"""
Created at 18.02.2021
"""
import numpy as np

from conmech.forces import Forces


class Solver:
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
        self.body_prop = body_prop
        self.contact_law = contact_law
        self.friction_bound = friction_bound

        self.mesh = mesh

        # Added
        self.time_step = time_step
        self.currentTime = 0
        self.u_vector = np.zeros([self.mesh.independent_nodes_count * 2])
        self.v_vector = np.zeros([self.mesh.independent_nodes_count * 2])

        self.const_elasticity = mesh.const_elasticity

        self.forces = Forces(mesh, inner_forces, outer_forces)
        self.forces.setF()

    def __str__(self):
        raise NotImplementedError()

    def iterate(self, velocity):
        self.v_vector = velocity.reshape(-1)
        self.u_vector = self.u_vector + self.time_step * self.v_vector

    def solve(self, initial_guess, **kwargs):
        raise NotImplementedError()
