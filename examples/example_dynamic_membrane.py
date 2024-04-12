from dataclasses import dataclass
from typing import Optional

import numpy as np
from conmech.helpers.config import Config
from conmech.mesh.boundaries_description import BoundariesDescription
from conmech.plotting.drawer import Drawer
from conmech.scenarios.problems import WaveProblem
from conmech.simulations.problem_solver import WaveSolver
from conmech.properties.mesh_description import CrossMeshDescription


@dataclass()
class MembraneSetup(WaveProblem):
    time_step: ... = 0.1

    @staticmethod
    def inner_forces(x: np.ndarray, t: Optional[float] = None) -> np.ndarray:
        return np.array([.2])

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
    runner = WaveSolver(setup, "direct")

    states = runner.solve(
        n_steps=32,
        output_step=(0, 32),
        initial_displacement=setup.initial_displacement,
        initial_velocity=setup.initial_velocity,
        verbose=True,
    )
    drawer = Drawer(state=states[-1], config=config)
    drawer.draw(
        show=config.show, save=config.save, foundation=False,
    )


if __name__ == "__main__":
    main(Config().init())
