"""
Created at 18.02.2021
"""

import numpy as np


class State:
    def __init__(self, body):
        self.body = body
        self.displacement: np.ndarray = np.zeros((self.body.mesh.nodes_count, 2))
        self.displaced_nodes: np.ndarray = np.copy(self.body.mesh.initial_nodes)
        self.velocity: np.ndarray = np.zeros((self.body.mesh.nodes_count, 2))
        self.time = 0

    def set_displacement(self, displacement_vector: np.ndarray, time: float = 0):
        self.displacement = displacement_vector.reshape((2, -1)).T
        self.displaced_nodes[: self.body.mesh.nodes_count, :2] = (
            self.body.mesh.initial_nodes[: self.body.mesh.nodes_count, :2]
            + self.displacement[:, :2]
        )
        self.time = time

    def set_velocity(
        self, velocity_vector: np.ndarray, time: float = 0, *, update_displacement: bool
    ):
        self.velocity = velocity_vector.reshape((2, -1)).T
        if update_displacement:
            dt = time - self.time
            self.displacement += dt * self.velocity
            self.displaced_nodes[: self.body.mesh.nodes_count, :2] = (
                self.body.mesh.initial_nodes[: self.body.mesh.nodes_count, :2]
                + self.displacement[:, :2]
            )
        self.time = time

    def __getitem__(self, item) -> np.ndarray:
        if item in (0, "displacement"):
            return self.displacement
        if item in (1, "velocity"):
            return self.velocity
        raise ValueError(f"Unknown coordinates {item}")

    def copy(self) -> "State":
        return self.__copy__()

    def __copy__(self) -> "State":
        copy = State(self.body)
        copy.displacement[:] = self.displacement
        copy.displaced_nodes[:] = self.displaced_nodes
        copy.velocity[:] = self.velocity
        copy.time = self.time
        return copy


class TemperatureState(State):
    def __init__(self, body):
        super().__init__(body)
        self.temperature = np.zeros(self.body.mesh.nodes_count)

    def set_temperature(self, temperature_vector: np.ndarray):
        self.temperature = temperature_vector

    def __copy__(self) -> "TemperatureState":
        copy = TemperatureState(self.body)
        copy.displacement[:] = self.displacement
        copy.displaced_nodes[:] = self.displaced_nodes
        copy.velocity[:] = self.velocity
        copy.time = self.time
        copy.temperature[:] = self.temperature
        return copy


class PiezoelectricState(State):
    def __init__(self, grid):
        super().__init__(grid)
        self.electric_potential = np.zeros(self.body.mesh.nodes_count)

    def set_electric_potential(self, electric_vector: np.ndarray):
        self.electric_potential = electric_vector

    def __copy__(self) -> "PiezoelectricState":
        copy = PiezoelectricState(self.body)
        copy.displacement[:] = self.displacement
        copy.displaced_nodes[:] = self.displaced_nodes
        copy.velocity[:] = self.velocity
        copy.time = self.time
        copy.electric_potential[:] = self.electric_potential
        return copy
