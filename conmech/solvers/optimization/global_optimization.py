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
        body_coeff,
        time_step,
        contact_law,
        friction_bound,
    ):
        super().__init__(
            grid,
            inner_forces,
            outer_forces,
            body_coeff,
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
        return self.B

    def get_right_hand_side(self):
        return self.forces.F_vector


@Solvers.register("quasistatic", "global", "global optimization")
class Quasistatic(Global):
    def __init__(
        self,
        mesh,
        inner_forces,
        outer_forces,
        body_coeff,
        time_step,
        contact_law,
        friction_bound,
    ):
        self.A = mesh.A
        super().__init__(
            mesh,
            inner_forces,
            outer_forces,
            body_coeff,
            time_step,
            contact_law,
            friction_bound,
        )

    def get_left_hand_side(self):
        return self.A

    def get_right_hand_side(self):
        return self.forces.F_vector - self.B @ self.u_vector.T

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
        body_coeff,
        time_step,
        contact_law,
        friction_bound,
    ):
        self.dim = mesh.dimension
        self.ACC = mesh.ACC
        self.K = mesh.K
        self.C2T = mesh.C2T
        self.t_vector = np.zeros(mesh.independent_nodes_count)
        super().__init__(
            mesh,
            inner_forces,
            outer_forces,
            body_coeff,
            time_step,
            contact_law,
            friction_bound,
        )

        self._point_temperature = (
            (1 / self.time_step)
            * self.mesh.ACC[
                : self.mesh.independent_nodes_count, : self.mesh.independent_nodes_count
            ]
            + self.K[
                : self.mesh.independent_nodes_count, : self.mesh.independent_nodes_count
            ]
        )

        self.Q = self.recalculate_temperature()

    @property
    def T(self):
        return self._point_temperature

    def get_left_hand_side(self):
        return self.A + (1 / self.time_step) * self.ACC

    def get_right_hand_side(self):
        X = -1 * self.B @ self.u_vector

        X += (1 / self.time_step) * self.ACC @ self.v_vector
        
        X += np.tile(self.t_vector, self.dim) @ self.C2T
        
        return self.forces.F_vector + X


    def iterate(self, velocity):
        super(Global, self).iterate(velocity)
        self._point_forces = self.recalculate_forces()
        self.Q = self.recalculate_temperature()

    def recalculate_temperature(self):
        C2X, C2Y = self.mesh.C2X, self.mesh.C2Y

        ind = self.mesh.independent_nodes_count
        C2Xv = C2X @ self.v_vector[0:ind]
        C2Yv = C2Y @ self.v_vector[ind : 2 * ind]

        Q1 = (1 / self.time_step) * self.ACC[:ind, :ind] @ self.t_vector

        QBig = Q1 - C2Xv - C2Yv

        return QBig
