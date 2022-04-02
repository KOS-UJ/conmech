"""
Created 22.02.2021
"""

import numpy as np

from conmech.helpers import nph
from conmech.solvers._solvers import Solvers
from conmech.solvers.optimization.optimization import Optimization


class Global(Optimization):
    def __init__(
            self,
            grid,
            inner_forces,
            outer_forces,
            body_prop,
            time_step,
            contact_law,
            friction_bound,
    ):
        super().__init__(
            grid,
            inner_forces,
            outer_forces,
            body_prop,
            time_step,
            contact_law,
            friction_bound,
        )
        self._point_relations = self.get_left_hand_side()
        self._point_forces = self.recalculate_forces()

    def __str__(self):
        return "global optimization"

    @property
    def point_relations(self) -> np.ndarray:
        return self._point_relations

    @property
    def point_forces(self) -> np.ndarray:
        return self._point_forces

    def recalculate_forces(self):
        return self.get_right_hand_side()

    def get_left_hand_side(self):
        raise NotImplementedError()

    def get_right_hand_side(self):
        raise NotImplementedError()


@Solvers.register("static", "global", "global optimization")
class Static(Global):
    def get_left_hand_side(self):
        return self.const_elasticity

    def get_right_hand_side(self):
        return self.forces.F_vector


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
        self.const_viscosity = mesh.const_viscosity
        super().__init__(
            mesh,
            inner_forces,
            outer_forces,
            body_prop,
            time_step,
            contact_law,
            friction_bound,
        )

    def get_left_hand_side(self):
        return self.const_viscosity

    def get_right_hand_side(self):
        return self.forces.F_vector - self.const_elasticity @ self.u_vector.T

    def iterate(self, velocity):
        super(Global, self).iterate(velocity)
        self._point_forces = self.recalculate_forces()


@Solvers.register("dynamic", "global", "global optimization")
class Dynamic(Quasistatic):
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
        self.dim = mesh.dimension
        self.ACC = mesh.ACC
        self.K = mesh.K
        self.C2T = mesh.C2T
        self.ind = mesh.independent_nodes_count
        self.t_vector = np.zeros(self.ind)
        super().__init__(
            mesh,
            inner_forces,
            outer_forces,
            body_prop,
            time_step,
            contact_law,
            friction_bound,
        )

        self._point_temperature = (1 / self.time_step) * self.mesh.ACC[
                                                         : self.ind, : self.ind
                                                         ] + self.K[: self.ind, : self.ind]

        self.Q = self.recalculate_temperature()

    @property
    def T(self):
        return self._point_temperature

    def get_left_hand_side(self):
        return self.const_viscosity + (1 / self.time_step) * self.ACC

    def get_right_hand_side(self):
        X = -1 * self.const_elasticity @ self.u_vector

        X += (1 / self.time_step) * self.ACC @ self.v_vector

        X += np.tile(self.t_vector, self.dim) @ self.C2T

        return self.forces.F_vector + X

    def iterate(self, velocity):
        super(Global, self).iterate(velocity)
        self._point_forces = self.recalculate_forces()
        self.Q = self.recalculate_temperature()

    def recalculate_temperature(self):
        X = (-1) * nph.unstack_and_sum_columns(self.C2T @ self.v_vector, dim=self.dim)

        X += (1 / self.time_step) * self.ACC[: self.ind, : self.ind] @ self.t_vector

        return X
