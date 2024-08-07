from dataclasses import dataclass
from typing import Optional

import numpy as np
from conmech.helpers.config import Config
from conmech.mesh.boundaries_description import BoundariesDescription
from conmech.plotting.drawer import Drawer
from conmech.scenarios.problems import PoissonProblem
from conmech.simulations.problem_solver import PoissonSolver
from conmech.properties.mesh_description import CrossMeshDescription


@dataclass()
class StaticPoissonSetup(PoissonProblem):
    @staticmethod
    def internal_temperature(x: np.ndarray, t: Optional[float] = None) -> np.ndarray:
        if 0.4 <= x[0] <= 0.6 and 0.4 <= x[1] <= 0.6:
            return np.array([-10.0])
        return np.array([2 * np.pi**2 * np.sin(np.pi * x[0]) * np.sin(np.pi * x[1])])

    @staticmethod
    def outer_temperature(
        x: np.ndarray, v: Optional[np.ndarray] = None, t: Optional[float] = None
    ) -> np.ndarray:
        if x[0] == 1:
            return np.array([10.0])
        return np.array([0.0])

    boundaries: ... = BoundariesDescription(
        dirichlet=(
            lambda x: x[1] == 0 or x[0] == 0 or x[1] == 1,
            lambda x: np.full(x.shape[0], 0),
        )
    )


def main(config: Config):
    """
    Entrypoint to example.

    To see result of simulation you need to call from python `main(Config().init())`.
    """
    mesh_descr = CrossMeshDescription(
        initial_position=None, max_element_perimeter=0.125, scale=[1, 1]
    )
    setup = StaticPoissonSetup(mesh_descr)
    runner = PoissonSolver(setup, "direct")

    state = runner.solve(verbose=True)
    max_ = max(max(state.temperature), 1)
    min_ = min(min(state.temperature), 0)
    drawer = Drawer(state=state, config=config)
    drawer.cmap = "plasma"
    drawer.field_name = "temperature"
    drawer.draw(
        show=config.show,
        save=config.save,
        foundation=False,
        field_max=max_,
        field_min=min_,
    )


if __name__ == "__main__":
    main(Config().init())
