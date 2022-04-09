"""
Created 22.02.2021
"""

import numpy as np

from conmech.dynamics.statement import (
    StaticStatement,
    QuasistaticStatement,
    DynamicStatement,
    TemperatureStatement,
)
from conmech.solvers._solvers import Solvers
from conmech.solvers.optimization.optimization import Optimization


class Global(Optimization):
    def __init__(
        self,
        mesh,
        statement,
        inner_forces,
        outer_forces,
        body_prop,
        time_step,
        contact_law,
        friction_bound,
    ):
        super().__init__(
            mesh,
            statement,
            inner_forces,
            outer_forces,
            body_prop,
            time_step,
            contact_law,
            friction_bound,
        )

    def __str__(self):
        return "global optimization"

    @property
    def point_relations(self) -> np.ndarray:
        return self.statement.left_hand_side

    @property
    def point_forces(self) -> np.ndarray:
        return self.statement.right_hand_side


@Solvers.register("static", "global", "global optimization")
class Static(Global):
    def __init__(
        self, mesh, inner_forces, outer_forces, body_prop, time_step, contact_law, friction_bound
    ):
        self.statement = StaticStatement(mesh)
        super().__init__(
            mesh,
            self.statement,
            inner_forces,
            outer_forces,
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
        inner_forces,
        outer_forces,
        body_prop,
        time_step,
        contact_law,
        friction_bound,
    ):
        self.statement = QuasistaticStatement(mesh)
        super().__init__(
            mesh,
            self.statement,
            inner_forces,
            outer_forces,
            body_prop,
            time_step,
            contact_law,
            friction_bound,
        )

    def iterate(self, velocity):
        super().iterate(velocity)
        self.statement.update(displacement=self.u_vector)


@Solvers.register("dynamic", "global", "global optimization")
class Dynamic(Global):
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
        self.statement = DynamicStatement(mesh)
        self.temperature_statement = TemperatureStatement(mesh)
        super().__init__(
            mesh,
            self.statement,
            inner_forces,
            outer_forces,
            body_prop,
            time_step,
            contact_law,
            friction_bound,
        )
        self.temperature_statement.update(
            velocity=self.v_vector, temperature=self.t_vector, time_step=self.time_step
        )

    @property
    def node_temperature(self):
        return self.temperature_statement.left_hand_side

    @property
    def temperature_rhs(self):
        return self.temperature_statement.right_hand_side

    def iterate(self, velocity):
        super().iterate(velocity)
        self.statement.update(
            displacement=self.u_vector,
            velocity=self.v_vector,
            temperature=self.t_vector,
            time_step=self.time_step,
        )
        self.temperature_statement.update(
            velocity=self.v_vector, temperature=self.t_vector, time_step=self.time_step
        )
