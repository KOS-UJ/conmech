import pickle
from dataclasses import dataclass
from typing import Optional, Type

import numpy as np
from conmech.helpers.config import Config
from conmech.mesh.boundaries_description import BoundariesDescription
from conmech.plotting.drawer import Drawer
from conmech.scenarios.problems import PoissonProblem, ContactLaw
from conmech.simulations.problem_solver import PoissonSolver
from conmech.properties.mesh_description import CrossMeshDescription


def make_slope_contact_law(slope: float) -> Type[ContactLaw]:
    class TarziaContactLaw(ContactLaw):
        @staticmethod
        def potential_normal_direction(u_nu: float) -> float:
            b = 5
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
    contact_law: ... = make_slope_contact_law(slope=1000)

    @staticmethod
    def internal_temperature(x: np.ndarray, t: Optional[float] = None) -> np.ndarray:
        if 0.4 <= x[0] <= 0.6 and 0.4 <= x[1] <= 0.6:
            return np.array([-10.0])
        return np.array([2 * np.pi**2 * np.sin(np.pi * x[0]) * np.sin(np.pi * x[1])])

    @staticmethod
    def outer_temperature(x: np.ndarray, t: Optional[float] = None) -> np.ndarray:
        if x[1] > 0.5:
            return np.array([1.0])
        return np.array([-1.0])

    boundaries: ... = BoundariesDescription(
        dirichlet=(
            lambda x: x[0] == 0.0,
            lambda x: np.full(x.shape[0], 5),
        ),
        contact=lambda x: x[0] == 2.0,
    )


def main(config: Config):
    """
    Entrypoint to example.

    To see result of simulation you need to call from python `main(Config().init())`.
    """
    alphas = [0.01, 0.1, 1, 10, 100, 1000, 10000]
    ihs = [4, 8, 16, 32, 64, 128, 256]
    alphas = alphas
    ihs = ihs

    for alpha in alphas:
        for ih in ihs:
            try:
                if config.force:
                    simulate(config, alpha, ih)
                draw(config, alpha, ih)
            except FileNotFoundError:
                simulate(config, alpha, ih)
                draw(config, alpha, ih)


def simulate(config, alpha, ih):
    print(f"Simulate {alpha=}, {ih=}")
    mesh_descr = CrossMeshDescription(
        initial_position=None, max_element_perimeter=1 / ih, scale=[2, 1]
    )
    setup = StaticPoissonSetup(mesh_descr)
    setup.contact_law = make_slope_contact_law(slope=alpha)

    runner = PoissonSolver(setup, "global")

    state = runner.solve(verbose=True)

    if config.outputs_path:
        with open(
            f"{config.outputs_path}/alpha_{alpha}_ih_{ih}",
            "wb+",
        ) as output:
            # Workaround
            state.body.dynamics.force.outer.source = None
            state.body.dynamics.force.inner.source = None
            state.body.properties.relaxation = None
            state.setup = None
            state.constitutive_law = None
            pickle.dump(state, output)


def draw(config, alpha, ih):
    with open(f"{config.outputs_path}/alpha_{alpha}_ih_{ih}", "rb") as output:
        state = pickle.load(output)
    max_ = max(max(state.temperature), 1)
    min_ = min(min(state.temperature), 0)
    drawer = Drawer(state=state, config=config)
    drawer.cmap = "plasma"
    drawer.field_name = "temperature"
    drawer.draw(
        show=config.show, save=config.save, foundation=False, field_max=max_, field_min=min_
    )


if __name__ == "__main__":
    main(Config(outputs_path="./output/BOT2023", force=False).init())
