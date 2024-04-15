from dataclasses import dataclass
from typing import Optional

import numpy as np
from conmech.helpers.config import Config
from conmech.mesh.boundaries_description import BoundariesDescription
from conmech.plotting.drawer import Drawer
from conmech.scenarios.problems import ContactWaveProblem, ContactLaw, \
    InteriorContactLaw
from conmech.simulations.problem_solver import WaveSolver
from conmech.properties.mesh_description import CrossMeshDescription


class DampedNormalCompliance(InteriorContactLaw):
    @staticmethod
    def general_contact_condition(u, v):
        k = 50
        obstacle_level = 0.1
        beta = 15
        if u < obstacle_level:
            return 0
        return k * (u - obstacle_level) + beta * v


@dataclass()
class MembraneSetup(ContactWaveProblem):
    time_step: ... = 0.1
    contact_law: ... = DampedNormalCompliance()

    @staticmethod
    def inner_forces(
            x: np.ndarray,
            t: Optional[float] = None
    ) -> np.ndarray:
        return np.array([5])

    @staticmethod
    def outer_forces(
        x: np.ndarray, v: Optional[np.ndarray] = None, t: Optional[float] = None
    ) -> np.ndarray:
        return np.array([0.0])

    boundaries: ... = BoundariesDescription(
        dirichlet=lambda x: x[0] in (0, 1) or x[1] in (0, 1)
    )


def main(config: Config):
    """
    Entrypoint to example.

    To see result of simulation you need to call from python `main(Config().init())`.
    """
    mesh_descr = CrossMeshDescription(
        initial_position=None, max_element_perimeter=0.125, scale=[1, 1]
    )
    setup = MembraneSetup(mesh_descr)
    runner = WaveSolver(setup, "global")

    steps = 26

    states = runner.solve(
        n_steps=steps,
        output_step=range(0, 26),
        initial_displacement=setup.initial_displacement,
        initial_velocity=setup.initial_velocity,
        verbose=True,
        method="Powell",
    )
    for i, state in enumerate(states):
        print("97:", (state.displacement[97, 0]), (state.velocity[97, 0]), "50:", (state.displacement[50, 0]), (state.velocity[50, 0]))
        if i % 5 != 0:
            continue
        field = np.zeros(state.displacement.shape[0] * 2)
        zeros = np.zeros(state.displacement.shape[0] * 2)
        field[:field.shape[0] // 2] = state.displacement[:, 0] * 1
        field[field.shape[0] // 2:] = state.displacement[:, 0] * 1
        state.set_displacement(zeros, 0)
        state.temperature = field[:field.shape[0] // 2]
        drawer = Drawer(state=state, config=config)
        drawer.cmap = "plasma"
        drawer.field_name = "temperature"
        drawer.field_label = "displacement"
        drawer.draw(
            show=config.show, save=config.save, foundation=False,
        )
        field = np.zeros(state.velocity.shape[0] * 2)
        field[:field.shape[0] // 2] = state.velocity[:, 0] * 1
        field[field.shape[0] // 2:] = state.velocity[:, 0] * 1
        state.set_displacement(zeros, 0)
        state.temperature = field[:field.shape[0] // 2]
        drawer = Drawer(state=state, config=config)
        drawer.cmap = "plasma"
        drawer.field_name = "temperature"
        drawer.field_label = "velocity"
        drawer.draw(
            show=config.show, save=config.save, foundation=False, field_min=min(state.temperature), field_max=max(state.temperature)
        )


if __name__ == "__main__":
    main(Config().init())
