"""
Created 22.02.2021
"""

import numpy as np

from conmech.dynamics.statement import Variables
from conmech.solvers._solvers import SolversRegistry
from conmech.solvers.optimization.optimization import Optimization


class GlobalOptimization(Optimization):
    def __str__(self):
        return "global optimization"

    @property
    def node_relations(self) -> np.ndarray:
        return self.statement.left_hand_side

    @property
    def node_forces(self) -> np.ndarray:
        return self.statement.right_hand_side


@SolversRegistry.register("static", "global", "global optimization")
class StaticGlobalOptimization(GlobalOptimization):
    pass


@SolversRegistry.register("quasistatic", "global", "global optimization")
class QuasistaticGlobalOptimization(GlobalOptimization):
    def iterate(self, velocity):
        super().iterate(velocity)
        self.statement.update(Variables(displacement=self.u_vector))


@SolversRegistry.register("dynamic", "global", "global optimization")
class DynamicGlobalOptimization(GlobalOptimization):
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
