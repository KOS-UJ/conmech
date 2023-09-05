from dataclasses import dataclass
from typing import Optional, Type

import numpy as np
from conmech.helpers.config import Config
from conmech.mesh.boundaries_description import BoundariesDescription
from conmech.plotting.drawer import Drawer
from conmech.scenarios.problems import PoissonProblem, ContactLaw
from conmech.simulations.problem_solver import PoissonSolver


def make_slope_contact_law(slope: float) -> Type[ContactLaw]:
    class TarziaContactLaw(ContactLaw):
        @staticmethod
        def potential_normal_direction(u_nu: float) -> float:
            b = 30
            r = u_nu
            # EXAMPLE 11
            # if r < b:
            #     result = (r - b) ** 2
            # else:
            #     result = 1 - np.exp(-(r-b))
            # EXAMPLE 13
            result = 0.5 * (r - b) ** 2
            result *= slope
            return result

        @staticmethod
        def subderivative_normal_direction(u_nu: float, v_nu: float) -> float:
            raise NotImplementedError()

        @staticmethod
        def regularized_subderivative_tangential_direction(
            u_tau: np.ndarray, v_tau: np.ndarray, rho=1e-7
        ) -> float:
            """
            Coulomb regularization
            """
            raise NotImplementedError()

    return TarziaContactLaw


@dataclass()
class StaticPoissonSetup(PoissonProblem):
    grid_height: ... = 1
    elements_number: ... = (8, 8)

    contact_law: ... = make_slope_contact_law(slope=100)

    @staticmethod
    def internal_temperature(x: np.ndarray, t: Optional[float] = None) -> np.ndarray:
        if 0.4 <= x[0] <= 0.6 and 0.4 <= x[1] <= 0.6:
            return np.array([-10.0])
        return np.array([2 * np.pi ** 2 * np.sin(np.pi * x[0]) * np.sin(np.pi * x[1])])

    @staticmethod
    def outer_temperature(
            x: np.ndarray, v: Optional[np.ndarray] = None, t: Optional[float] = None
    ) -> np.ndarray:
        alpha = 1
        if x[0] == 1:
            result = np.full(1, alpha)
            return result
        return np.array([0.0])

    boundaries: ... = BoundariesDescription(
        dirichlet=(
            lambda x: x[1] == 0 or x[0] == 0 or x[1] == 1,
            lambda x: np.full(x.shape[0], 0),
        ),
        contact=lambda x: x[0] == 1,
    )


def main(config: Config):
    """
    Entrypoint to example.

    To see result of simulation you need to call from python `main(Config().init())`.
    """
    setup = StaticPoissonSetup(mesh_type="cross")
    runner = PoissonSolver(setup, "global")

    state = runner.solve(verbose=True)
    max_ = max(max(state.temperature), 1)
    min_ = min(min(state.temperature), 0)
    drawer = Drawer(state=state, config=config)
    drawer.cmap = "plasma"
    drawer.field_name = "temperature"
    drawer.draw(
        show=config.show, save=config.save, foundation=False, field_max=max_, field_min=min_
    )


if __name__ == "__main__":
    main(Config().init())
