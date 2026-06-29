import pickle
from dataclasses import dataclass
from typing import Optional, Type

import numpy as np
from conmech.helpers.config import Config
from conmech.mesh.boundaries_description import BoundariesDescription
from conmech.plotting.drawer import Drawer
from conmech.scenarios.problems import PoissonProblem
from conmech.dynamics.contact.contact_law import ContactLaw, PotentialOfContactLaw
from conmech.simulations.problem_solver import PoissonSolver
from conmech.properties.mesh_description import RectangleMeshDescription


B_COEF = 5

def make_slope_contact_law(slope: float) -> Type[ContactLaw]:
    class TarziaContactLaw(PotentialOfContactLaw):
        @staticmethod
        def potential_normal_direction(
            var_nu: float, static_displacement_nu: float, dt: float
        ) -> float:
            b = B_COEF
            r = var_nu
            # EXAMPLE 11
            if r < b:
                result = 0.5 * (r - b) ** 2
            else:
                result = np.log((r - b) + 1)
            result *= slope
            return result

        @staticmethod
        def subderivative_normal_direction(
            var_nu: float, static_displacement_nu: float, dt: float
        ) -> float:
            b = B_COEF
            r = var_nu
            # EXAMPLE 11
            if r < b:
                result = r - b
            else:
                result = 1 / (r - b  +1)
            result *= slope
            return result

    return TarziaContactLaw


@dataclass()
class StaticPoissonSetup(PoissonProblem):
    contact_law_2: ... = make_slope_contact_law(slope=1000)

    @staticmethod
    def internal_temperature(x: np.ndarray, t: Optional[float] = None) -> np.ndarray:
        return np.array([-4])

    @staticmethod
    def outer_temperature(x: np.ndarray, t: Optional[float] = None) -> np.ndarray:
        _y = x[1]
        return np.array([_y * (_y - 1) * 32])

    boundaries: ... = BoundariesDescription(
        dirichlet=(
            lambda x: x[1] == 0.0 or x[1] == 1.0,
            lambda x: np.full(x.shape[0], 5),
        ),
        # contact=lambda x: x[1] == 1.0,
    )


def main(config: Config):
    """
    Entrypoint to example.

    To see result of simulation you need to call from python `main(Config().init())`.
    """
    alphas = [0.01, 0.1, 1, 10, 100, 1000, 10_000, 1_000_000, 1_000_000_000, np.inf]
    ihs = [4, 8, 16, 32, 48, 72]
    alphas = alphas if not config.test else alphas[:1]
    ihs = ihs if not config.test else ihs[:1]

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
    mesh_descr = RectangleMeshDescription(
        initial_position=None, max_element_perimeter=1 / ih, scale=[2, 1]
    )
    setup = StaticPoissonSetup(mesh_descr)
    setup.contact_law_2 = make_slope_contact_law(slope=alpha)

    solving_method = "schur" if alpha != np.inf else "direct"
    runner = PoissonSolver(setup, solving_method)

    state = runner.solve(verbose=True, method="qsm")

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
    drawer.deformed_mesh_color = None
    drawer.original_mesh_color = None

    drawer.draw(
        title=f"alpha={alpha}, ih={ih}",
        show=config.show,
        save=not config.show,
        foundation=False,
        field_max=max_,
        field_min=min_,
    )


if __name__ == "__main__":
    main(Config(outputs_path="./output/BOT2023", force=False).init())
