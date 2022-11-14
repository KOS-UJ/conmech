"""
Created at 18.02.2021
"""
from typing import Callable, Optional

import numpy as np

from conmech.dynamics.statement import Statement, Variables
from conmech.scenarios.problems import ContactLaw
from conmech.scene.body_forces import BodyForces


class Solver:
    def __init__(
        self,
        statement: Statement,
        body: BodyForces,
        time_step: float,
        contact_law: Optional[ContactLaw],
        friction_bound: Optional[Callable[[float], float]],
    ):
        self.contact_law: Optional[ContactLaw] = contact_law
        self.friction_bound: Optional[Callable[[float], float]] = friction_bound

        self.body: BodyForces = body
        self.statement: Statement = statement

        self.time_step: float = time_step
        self.current_time: float = 0
        self.b_vector = np.zeros(self.body.mesh.nodes_count * 2)
        self.u_vector: np.ndarray = np.zeros(self.body.mesh.independent_nodes_count * 2)
        self.v_vector: np.ndarray = np.zeros(self.body.mesh.independent_nodes_count * 2)
        self.t_vector: np.ndarray = np.zeros(self.body.mesh.independent_nodes_count)
        self.p_vector: np.ndarray = np.zeros(self.body.mesh.independent_nodes_count)  # TODO #23

        self.elasticity: np.ndarray = body.elasticity

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

    def __str__(self) -> str:
        raise NotImplementedError()

    def iterate(self):
        pass

    def _solve_impl(
        self, initial_guess, *, velocity: np.ndarray, displacement: np.ndarray, **kwargs
    ):
        raise NotImplementedError()

    def solve(self, initial_guess: np.ndarray, **kwargs) -> np.ndarray:
        solution = self._solve_impl(
            initial_guess, velocity=self.v_vector, displacement=self.u_vector, **kwargs
        )

        for dirichlet_cond in self.statement.find_dirichlet_conditions():
            c = self.body.mesh.boundaries.boundaries[dirichlet_cond].node_condition
            node_count = self.body.mesh.nodes_count
            for i, j in self.body.mesh.boundaries.get_all_boundary_indices(
                dirichlet_cond, node_count, self.statement.dimension
            ):
                solution[i] = c[j]

        return solution
