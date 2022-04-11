"""
Created 22.02.2021
"""

import numpy as np

from conmech.dynamics.statement import (
    StaticDisplacementStatement,
    QuasistaticVelocityStatement,
    DynamicVelocityStatement,
    TemperatureStatement,
    Variables, DynamicVelocityWithTemperatureStatement,
)
from conmech.solvers._solvers import Solvers
from conmech.solvers.optimization.optimization import Optimization


class Global(Optimization):
    def __str__(self):
        return "global optimization"

    @property
    def node_relations(self) -> np.ndarray:
        return self.statement.left_hand_side

    @property
    def node_forces(self) -> np.ndarray:
        return self.statement.right_hand_side


@Solvers.register("static", "global", "global optimization")
class Static(Global):
    def __init__(self, mesh, body_prop, time_step, contact_law, friction_bound):
        statement = StaticDisplacementStatement(mesh)
        super().__init__(
            mesh,
            statement,
            body_prop,
            time_step,
            contact_law,
            friction_bound,
        )


@Solvers.register("quasistatic", "global", "global optimization")
class Quasistatic(Global):
    def __init__(
        self,
        mesh,
        body_prop,
        time_step,
        contact_law,
        friction_bound,
    ):
        statement = QuasistaticVelocityStatement(mesh)
        super().__init__(
            mesh,
            statement,
            body_prop,
            time_step,
            contact_law,
            friction_bound,
        )

    def iterate(self, velocity):
        super().iterate(velocity)
        self.statement.update(Variables(displacement=self.u_vector))


@Solvers.register("dynamic", "global", "global optimization")
class Dynamic(Global):
    def __init__(
        self,
        mesh,
        body_prop,
        time_step,
        contact_law,
        friction_bound,
    ):
        statement = DynamicVelocityWithTemperatureStatement(mesh)
        self.temperature_statement = TemperatureStatement(mesh)
        super().__init__(
            mesh,
            statement,
            body_prop,
            time_step,
            contact_law,
            friction_bound,
        )
        self.temperature_statement.update(
            Variables(
                velocity=self.v_vector,
                temperature=self.t_vector,
                time_step=self.time_step,
            )
        )

    @property
    def node_temperature(self):
        return self.temperature_statement.left_hand_side

    @property
    def temper_rhs(self):
        return self.temperature_statement.right_hand_side

    def iterate(self, velocity):
        super().iterate(velocity)
        self.statement.update(
            Variables(
                displacement=self.u_vector,
                velocity=self.v_vector,
                temperature=self.t_vector,
                time_step=self.time_step,
            )
        )
        self.temperature_statement.update(
            Variables(
                velocity=self.v_vector,
                temperature=self.t_vector,
                time_step=self.time_step,
            )
        )
