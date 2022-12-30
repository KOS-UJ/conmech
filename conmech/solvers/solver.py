"""
Created at 18.02.2021
"""
import numpy as np

from conmech.dynamics.statement import Statement, Variables


class Solver:
    def __init__(
        self,
        statement,
        body,
        time_step,
        contact_law,
        friction_bound,
    ):
        self.contact_law = contact_law
        self.friction_bound = friction_bound

        self.body = body
        self.statement: Statement = statement

        self.time_step = time_step
        self.current_time = 0
        self.u_vector = np.zeros(self.body.nodes_count * 2)
        self.v_vector = np.zeros(self.body.nodes_count * 2)
        self.t_vector = np.zeros(self.body.nodes_count)
        self.p_vector = np.zeros(self.body.nodes_count)  # TODO #23

        self.elasticity = body.matrices.elasticity

        self.statement.update(
            Variables(
                displacement=self.u_vector,
                velocity=self.v_vector,
                temperature=self.t_vector,
                time_step=self.time_step,
                electric_potential=self.p_vector,
            )
        )

    def __str__(self):
        raise NotImplementedError()

    def iterate(self, velocity):
        self.v_vector[:] = velocity.reshape(-1)
        self.u_vector[:] = self.u_vector + self.time_step * self.v_vector

    def _solve_impl(
        self, initial_guess, *, velocity: np.ndarray, displacement: np.ndarray, **kwargs
    ):
        raise NotImplementedError()

    def solve(self, initial_guess: np.ndarray, **kwargs) -> np.ndarray:
        solution = self._solve_impl(
            initial_guess, velocity=self.v_vector, displacement=self.u_vector, **kwargs
        )

        for dirichlet_cond in self.statement.find_dirichlet_conditions():
            c = self.body.boundaries.boundaries[dirichlet_cond].node_condition
            node_count = self.body.nodes_count
            for i, j in self.body.boundaries.get_all_boundary_indices(
                dirichlet_cond, node_count, self.statement.dimension
            ):
                solution[i] = c[j]

        return solution
