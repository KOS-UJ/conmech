"""
Created 22.02.2021
"""

import numpy as np

from conmech.dynamics.statement import (
    Variables,
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
    pass


@Solvers.register("quasistatic", "global", "global optimization")
class Quasistatic(Global):
    def iterate(self, velocity):
        super().iterate(velocity)
        self.statement.update(
            Variables(
                displacement=self.u_vector,
                electric_potential=self.p_vector,
            )
        )


@Solvers.register("dynamic", "global", "global optimization")
class Dynamic(Global):
    def iterate(self, velocity):
        super().iterate(velocity)
        self.statement.update(
            Variables(
                displacement=self.u_vector,
                velocity=self.v_vector,
                temperature=self.t_vector,
                electric_potential=self.p_vector,
                time_step=self.time_step,
            )
        )
