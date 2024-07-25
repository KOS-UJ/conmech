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
    def lhs(self) -> np.ndarray:
        return self.statement.left_hand_side.data

    @property
    def rhs(self) -> np.ndarray:
        return self.statement.right_hand_side


@SolversRegistry.register("static", "global", "global optimization")
class StaticGlobalOptimization(GlobalOptimization):
    pass


@SolversRegistry.register("quasistatic", "global", "global optimization")
class QuasistaticGlobalOptimization(GlobalOptimization):
    def iterate(self):
        self.statement.update(
            Variables(
                displacement=self.u_vector,
                electric_potential=self.p_vector,
                time_step=self.time_step,
                time=self.current_time,
            )
        )


@SolversRegistry.register("quasistatic relaxation", "global", "global optimization")
class QuasistaticRelaxedGlobalOptimization(GlobalOptimization):
    def iterate(self):
        self.statement.update(
            Variables(
                absement=self.b_vector,
                displacement=self.u_vector,
                time_step=self.time_step,
            )
        )


@SolversRegistry.register("dynamic", "global", "global optimization")
class DynamicGlobalOptimization(GlobalOptimization):
    def iterate(self):
        self.statement.update(
            Variables(
                displacement=self.u_vector,
                velocity=self.v_vector,
                temperature=self.t_vector,
                electric_potential=self.p_vector,
                time_step=self.time_step,
                time=self.current_time,
            )
        )
