"""
Created at 18.02.2021
"""
from abc import ABC
from typing import Callable, Optional

import numpy as np

from conmech.dynamics.statement import Statement, Variables
from conmech.scenarios.problems import ContactLaw
from conmech.scene.body_forces import BodyForces


class Solver(ABC):
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
        self.u_vector: np.ndarray = np.zeros(self.body.mesh.independent_nodes_count * 2)
        self.v_vector: np.ndarray = np.zeros(self.body.mesh.independent_nodes_count * 2)
        self.t_vector: np.ndarray = np.zeros(self.body.mesh.independent_nodes_count)
        self.p_vector: np.ndarray = np.zeros(self.body.mesh.independent_nodes_count)  # TODO #23

        self.elasticity: np.ndarray = body.elasticity

        self.statement.update(
            Variables(
                displacement=self.u_vector,
                velocity=self.v_vector,
                temperature=self.t_vector,
                time_step=self.time_step,
                electric_potential=self.p_vector,
            )
        )

    def __str__(self) -> str:
        raise NotImplementedError()

    def iterate(self, velocity: np.ndarray) -> None:
        self.v_vector = velocity.reshape(-1)
        self.u_vector = self.u_vector + self.time_step * self.v_vector

    def solve(self, initial_guess: np.ndarray, *, velocity: np.ndarray, **kwargs) -> np.ndarray:
        raise NotImplementedError()
