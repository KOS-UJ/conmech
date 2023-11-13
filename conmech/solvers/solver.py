"""
Created at 18.02.2021
"""

from typing import Optional

import numpy as np
import time

from conmech.dynamics.statement import Statement, Variables
from conmech.dynamics.contact.contact_law import ContactLaw


class Solver:
    def __init__(
        self,
        statement: Statement,
        body: "Body",
        time_step: float,
        contact_law: Optional[ContactLaw] = None,
        driving_vector: bool = False,
    ):
        self.contact_law: Optional[ContactLaw] = contact_law

        self.body = body
        self.statement: Statement = statement

        self.time_step: float = time_step
        self.current_time: float = 0
        self.b_vector = np.zeros(self.body.mesh.nodes_count * self.body.mesh.dimension)
        self.u_vector: np.ndarray = np.zeros(self.body.mesh.nodes_count * self.body.mesh.dimension)
        self.v_vector: np.ndarray = np.zeros(self.body.mesh.nodes_count * self.body.mesh.dimension)
        self.t_vector: np.ndarray = np.zeros(self.body.mesh.nodes_count)
        self.p_vector: np.ndarray = np.zeros(self.body.mesh.nodes_count)  # TODO #23

        self.elasticity: np.ndarray = body.dynamics.elasticity

        self.driving_vector = driving_vector

        self.statement.update(
            Variables(
                absement=self.b_vector,
                displacement=self.u_vector,
                velocity=self.v_vector,
                temperature=self.t_vector,
                time_step=self.time_step,
                time=self.current_time,
                electric_potential=self.p_vector,
            )
        )

        self.last_timing = None

    def __str__(self) -> str:
        raise NotImplementedError()

    def iterate(self):
        pass

    def _solve_impl(
        self,
        initial_guess,
        *,
        variable_old: np.ndarray,
        displacement: np.ndarray,
        **kwargs,
    ):
        raise NotImplementedError()

    def solve(self, initial_guess: np.ndarray, **kwargs) -> np.ndarray:
        start = time.time()
        solution = self._solve_impl(
            initial_guess,
            variable_old=self.v_vector,
            displacement=self.u_vector,
            **kwargs,  # TODO: FIXME variable_old
        )

        for dirichlet_cond in self.statement.find_dirichlet_conditions():
            c = self.body.mesh.boundaries.boundaries[dirichlet_cond].node_condition
            node_count = self.body.mesh.nodes_count
            for i, j in self.body.mesh.boundaries.get_all_boundary_indices(
                dirichlet_cond, node_count, self.statement.dimension_in
            ):
                solution[i] = c[j]

        self.last_timing = time.time() - start

        return solution
