"""
Created 22.02.2021
"""

import numpy as np

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
        return self.elasticity

    def get_right_hand_side(self):
        return self.forces.forces_vector


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
        self.viscosity = mesh.viscosity
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
        return self.viscosity

    def get_right_hand_side(self):
        return self.forces.forces_vector - self.elasticity @ self.u_vector.T

    def iterate(self, velocity):
        super().iterate(velocity)
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
        self.acceleration_operator = mesh.acceleration_operator
        self.thermal_conductivity = mesh.thermal_conductivity
        self.thermal_expansion = mesh.thermal_expansion
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

        self._point_temperature = (1 / self.time_step) * self.mesh.acceleration_operator[
                                                         : self.ind, : self.ind
                                                         ] + self.thermal_conductivity[: self.ind,
                                                             : self.ind]

        self.temperature_rhs = self.recalculate_temperature()

    @property
    def node_temperature(self):
        return self._point_temperature

    def get_left_hand_side(self):
        return self.viscosity + (1 / self.time_step) * self.acceleration_operator

    def get_right_hand_side(self):
        A = -1 * self.elasticity @ self.u_vector

        A += (1 / self.time_step) * self.acceleration_operator @ self.v_vector

        A += self.thermal_expansion.T @ self.t_vector

        return self.forces.forces_vector + A

    def iterate(self, velocity):
        super().iterate(velocity)
        self._point_forces = self.recalculate_forces()
        self.temperature_rhs = self.recalculate_temperature()

    def recalculate_temperature(self):
        A = (-1) * self.thermal_expansion @ self.v_vector

        A += (1 / self.time_step) * \
             self.acceleration_operator[: self.ind, : self.ind] @ self.t_vector

        return A
