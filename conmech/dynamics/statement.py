import numpy as np


class Statement:
    def __init__(self, dynamics):
        self.dynamics = dynamics
        self.left_hand_side = None
        self.right_hand_side = None

    def update_left_hand_side(self, **kwargs):
        raise NotImplementedError()

    def update_right_hand_side(self, **kwargs):
        raise NotImplementedError()

    def update(self, **kwargs):
        self.update_left_hand_side(**kwargs)
        self.update_right_hand_side(**kwargs)


class StaticStatement(Statement):
    def update_left_hand_side(self, **kwargs):
        self.left_hand_side = self.dynamics.elasticity

    def update_right_hand_side(self, **kwargs):
        self.right_hand_side = self.dynamics.forces.forces_vector


class QuasistaticStatement(Statement):
    def update_left_hand_side(self, **kwargs):
        self.left_hand_side = self.dynamics.viscosity

    def update_right_hand_side(self, displacement: np.ndarray, **kwargs):
        self.right_hand_side = (
            self.dynamics.forces.forces_vector - self.dynamics.elasticity @ displacement.T
        )


class DynamicStatement(Statement):
    def update_left_hand_side(self, time_step: float, **kwargs):
        self.left_hand_side = (
            self.dynamics.viscosity + (1 / time_step) * self.dynamics.acceleration_operator
        )

    def update_right_hand_side(
        self,
        displacement: np.ndarray,
        velocity: np.ndarray,
        temperature: np.ndarray,
        time_step: float,
        **kwargs,
    ):
        A = -1 * self.dynamics.elasticity @ displacement

        A += (1 / time_step) * self.dynamics.acceleration_operator @ velocity

        A += self.dynamics.thermal_expansion.T @ temperature

        self.right_hand_side = self.dynamics.forces.forces_vector + A


class TemperatureStatement(Statement):
    def update_left_hand_side(self, time_step: float, **kwargs):
        ind = self.dynamics.independent_nodes_count

        self.left_hand_side = (1 / time_step) * self.dynamics.acceleration_operator[
            :ind, :ind
        ] + self.dynamics.thermal_conductivity[:ind, :ind]

    def update_right_hand_side(
        self, velocity: np.ndarray, temperature: np.ndarray, time_step: float, **kwargs
    ):
        rhs = (-1) * self.dynamics.thermal_expansion @ velocity

        ind = self.dynamics.independent_nodes_count

        rhs += (1 / time_step) * self.dynamics.acceleration_operator[:ind, :ind] @ temperature
        self.right_hand_side = rhs
        # self.right_hand_side = self.inner_temperature.F[:, 0] + Q1 - C2Xv - C2Yv  # TODO #50
