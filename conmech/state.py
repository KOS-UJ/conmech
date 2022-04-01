"""
Created at 18.02.2021
"""

import numpy as np


class State:
    def __init__(self, mesh):
        self.mesh = mesh
        self.displacement: np.ndarray = np.zeros((self.mesh.independent_nodes_count, 2))
        self.displaced_points: np.ndarray = np.copy(self.mesh.initial_nodes)
        self.velocity: np.ndarray = np.zeros((self.mesh.independent_nodes_count, 2))
        self.time = 0

    def set_displacement(self, displacement_vector: np.ndarray, t: float = 0):
        self.displacement = displacement_vector.reshape((2, -1)).T
        self.displaced_points[: self.mesh.independent_nodes_count, :2] = (
                self.mesh.initial_nodes[: self.mesh.independent_nodes_count, :2]
                + self.displacement[:, :2]
        )
        self.time = t

    def set_velocity(
            self, velocity_vector: np.ndarray, t: float = 0, *, update_displacement: bool
    ):
        self.velocity = velocity_vector.reshape((2, -1)).T
        if update_displacement:
            dt = t - self.time
            self.displacement += dt * self.velocity
            self.displaced_points[: self.mesh.independent_nodes_count, :2] = (
                    self.mesh.initial_nodes[: self.mesh.independent_nodes_count, :2]
                    + self.displacement[:, :2]
            )
        self.time = t

    def __getitem__(self, item) -> np.ndarray:
        if item in (0, "displacement"):
            return self.displacement
        if item in (1, "velocity"):
            return self.velocity

    def copy(self) -> "State":
        return self.__copy__()

    def __copy__(self) -> "State":
        copy = State(self.mesh)
        copy.displacement[:] = self.displacement
        copy.displaced_points[:] = self.displaced_points
        copy.velocity[:] = self.velocity
        copy.time = self.time
        return copy


class TemperatureState(State):
    def __init__(self, grid):
        super().__init__(grid)
        self.temperature = np.zeros(self.mesh.independent_nodes_count)

    def set_temperature(self, temperature_vector: np.ndarray):
        self.temperature = temperature_vector

    def __copy__(self) -> "TemperatureState":
        copy = TemperatureState(self.mesh)
        copy.displacement[:] = self.displacement
        copy.displaced_points[:] = self.displaced_points
        copy.velocity[:] = self.velocity
        copy.time = self.time
        copy.temperature[:] = self.temperature
        return copy
