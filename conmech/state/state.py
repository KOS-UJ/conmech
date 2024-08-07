"""
Created at 18.02.2021
"""

import pickle

import numpy as np


class State:
    def __init__(self, body):
        self.body = body
        self.body.state = self

        self.products = {}

        self.absement: np.ndarray = np.zeros((self.body.mesh.nodes_count, self.body.mesh.dimension))
        self.displacement: np.ndarray = np.zeros(
            (self.body.mesh.nodes_count, self.body.mesh.dimension)
        )
        self.displaced_nodes: np.ndarray = np.copy(self.body.mesh.nodes)
        self.velocity: np.ndarray = np.zeros((self.body.mesh.nodes_count, self.body.mesh.dimension))
        self.setup = None
        self.__stress: np.ndarray = None
        self.constitutive_law = None
        self.temperature = None  # TODO
        self.time: float = 0

    def set_displacement(
        self,
        displacement_vector: np.ndarray,
        time: float,
        *,
        update_absement: bool = False,
    ):
        self.displacement = displacement_vector.reshape((self.body.mesh.dimension, -1)).T
        self.displaced_nodes[: self.body.mesh.nodes_count, :] = (
            self.body.mesh.nodes[: self.body.mesh.nodes_count, :] + self.displacement[:, :]
        )
        if update_absement:
            dt = time - self.time
            self.absement += dt * self.displacement
        self.time = time
        self.update_products()

    def set_velocity(self, velocity_vector: np.ndarray, time: float, *, update_displacement: bool):
        self.velocity = velocity_vector.reshape((self.body.mesh.dimension, -1)).T
        if update_displacement:
            dt = time - self.time
            self.displacement += dt * self.velocity
            self.displaced_nodes[: self.body.mesh.nodes_count, :] = (
                self.body.mesh.nodes[: self.body.mesh.nodes_count, :] + self.displacement[:, :]
            )
        self.time = time
        self.update_products()

    @property
    def stress(self):
        if self.__stress is None and self.constitutive_law is not None:
            # pylint: disable=not-callable
            self.__stress = self.constitutive_law(
                velocity=self.velocity,
                displacement=self.displacement,
                absement=self.absement,
                setup=self.setup,
                elements=self.body.mesh.elements,
                nodes=self.body.mesh.nodes,
                time=self.time,
            )
        return self.__stress

    @property
    def stress_norm(self):
        return np.linalg.norm(self.stress, axis=(1, 2))

    @property
    def stress_x(self):
        return self.stress[:, 0, 0]

    @property
    def stress_y(self):
        return self.stress[:, 1, 1]

    def inject_product(self, product):
        self.products[product.name] = product

    def update_products(self):
        for prod in self.products.values():
            prod.update(self)

    def __getitem__(self, item: [int, str]) -> np.ndarray:
        if item in (0, "displacement"):
            return self.displacement
        if item in (1, "velocity"):
            return self.velocity
        raise ValueError(f"Unknown coordinates {item}")

    def copy(self) -> "State":
        # pylint: disable=unnecessary-dunder-call
        return self.__copy__()

    def __copy__(self) -> "State":
        copy = State(self.body)
        copy.absement[:] = self.absement
        copy.displacement[:] = self.displacement
        copy.displaced_nodes[:] = self.displaced_nodes
        copy.velocity[:] = self.velocity
        copy.time = self.time
        copy.products = {n: p.copy() for n, p in self.products.items()}
        return copy

    def save(self, path):
        fos = self.body.dynamics.force.outer.source
        self.body.dynamics.force.outer.source = None
        fis = self.body.dynamics.force.inner.source
        self.body.dynamics.force.inner.source = None
        s = self.setup
        self.setup = None
        cl = self.constitutive_law
        self.constitutive_law = None

        with open(path, "wb+") as file:
            pickle.dump(self, file)

        self.body.dynamics.force.outer.source = fos
        self.body.dynamics.force.inner.source = fis
        self.setup = s
        self.constitutive_law = cl

    @classmethod
    def load(cls, path):
        with open(path, "rb") as file:
            return pickle.load(file)


class TemperatureState(State):
    def __init__(self, body):
        super().__init__(body)
        self.temperature = np.zeros(self.body.mesh.nodes_count)

    def __getitem__(self, item: [int, str]) -> np.ndarray:
        if item == "temperature":
            return self.temperature
        return super().__getitem__(item)

    def set_temperature(self, temperature_vector: np.ndarray):
        self.temperature = temperature_vector

    def __copy__(self) -> "TemperatureState":
        copy = TemperatureState(self.body)
        copy.absement[:] = self.absement
        copy.displacement[:] = self.displacement
        copy.displaced_nodes[:] = self.displaced_nodes
        copy.velocity[:] = self.velocity
        copy.time = self.time
        copy.temperature[:] = self.temperature
        return copy


class PiezoelectricState(State):
    def __init__(self, body):
        super().__init__(body)
        self.electric_potential = np.zeros(self.body.mesh.nodes_count)

    def set_electric_potential(self, electric_vector: np.ndarray):
        self.electric_potential = electric_vector

    def __copy__(self) -> "PiezoelectricState":
        copy = PiezoelectricState(self.body)
        copy.absement[:] = self.absement
        copy.displacement[:] = self.displacement
        copy.displaced_nodes[:] = self.displaced_nodes
        copy.velocity[:] = self.velocity
        copy.time = self.time
        copy.electric_potential[:] = self.electric_potential
        return copy
