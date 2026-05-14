import pickle
from dataclasses import dataclass
from pathlib import Path
from typing import Optional, Type

import matplotlib.pyplot as plt
import matplotlib.tri as mtri
import numpy as np
from conmech.helpers.config import Config
from conmech.mesh.boundaries_description import BoundariesDescription
# Drawer is no longer used here; plotting is done directly with matplotlib.tri
from conmech.scenarios.problems import PoissonProblem
from conmech.dynamics.contact.contact_law import ContactLaw, PotentialOfContactLaw
from conmech.simulations.problem_solver import PoissonSolver
from conmech.properties.mesh_description import RectangleMeshDescription


B_COEF = 5
MAXD = 16#72
TO_PLOT = (
    ((np.inf, 4), (np.inf, MAXD)),
    ((10, MAXD), (100, MAXD)),
    ((0.1, MAXD), (1, MAXD)),
    ((0, 4), (0, MAXD)),
)

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
    contact_law_2: Type[ContactLaw] = make_slope_contact_law(slope=1000)

    @staticmethod
    def internal_temperature(x: np.ndarray, t: Optional[float] = None) -> np.ndarray:
        return np.array([-4])

    @staticmethod
    def outer_temperature(x: np.ndarray, t: Optional[float] = None) -> np.ndarray:
        _y = x[1]
        return np.array([_y * (_y - 1) * 32])

    boundaries: BoundariesDescription = BoundariesDescription(
        dirichlet=(
            lambda x: x[1] == 0.0 ,#or x[1] == 1.0,
            lambda x: np.full(x.shape[0], 5),
        ),
        contact=lambda x: x[1] == 1.0,
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
            print(f"Configuration: {alpha=}, {ih=}")
            load_or_simulate(config, alpha, ih, only_ensure=True)

    Path(config.outputs_path).mkdir(parents=True, exist_ok=True)
    to_plot = TO_PLOT if not config.test else TO_PLOT[:1]
    draw_grid(config, to_plot)


def state_path(config, alpha, ih) -> Path:
    return Path(config.outputs_path) / f"alpha_{alpha}_ih_{ih}"


def load_or_simulate(config, alpha, ih, only_ensure=False):
    path = state_path(config, alpha, ih)
    if config.force or not path.exists():
        simulate(config, alpha, ih)
    if only_ensure:
        return None
    with open(path, "rb") as output:
        return pickle.load(output)


def draw_grid(config, to_plot):
    rows = len(to_plot)
    cols = len(to_plot[0])
    fig, axes = plt.subplots(rows, cols, figsize=(7 * cols, 4.5 * rows), squeeze=False)

    # Do a single-pass scan to compute global field min/max without keeping states in memory.
    field_min = float("inf")
    field_max = float("-inf")
    for row in to_plot:
        for alpha, ih in row:
            print(f"Scanning {alpha=}, {ih=} for field range")
            state = load_or_simulate(config, alpha, ih)
            try:
                temperature = np.asarray(state.temperature, dtype=float).ravel()
                field_min = min(field_min, float(temperature.min().item()))
                field_max = max(field_max, float(temperature.max().item()))
            finally:
                # release heavy object before next iteration
                del state
                import gc

                gc.collect()

    # Now render each subplot, loading and discarding state one-by-one to limit RAM usage
    for row_idx, row in enumerate(to_plot):
        for col_idx, (alpha, ih) in enumerate(row):
            print(f"Rendering {alpha=}, {ih=}")
            state = load_or_simulate(config, alpha, ih)
            ax = axes[row_idx, col_idx]

            # build triangulation and plot field directly to avoid Drawer autoscaling issues
            nodes = state.body.mesh.nodes
            tris = state.body.mesh.elements
            vals = np.asarray(state.temperature).ravel()

            triang = mtri.Triangulation(nodes[:, 0], nodes[:, 1], tris)

            # contour lines and filled contour
            ax.tricontour(nodes[:, 0], nodes[:, 1], tris, vals, 15, colors="k", linewidths=0.2)
            n_layers = 100
            cf = ax.tricontourf(
                nodes[:, 0],
                nodes[:, 1],
                tris,
                vals,
                n_layers,
                cmap="plasma",
                vmin=field_min,
                vmax=field_max,
            )

            # mesh outline
            ax.triplot(triang, "k-", alpha=0.15, linewidth=0.3)

            inf_symbol = r"$\infty$"
            ax.set_title(fr"$\alpha$={alpha if alpha != np.inf else inf_symbol}, h=1/{ih}")
            ax.set_aspect("equal", adjustable="box")
            # set axis limits to nodes extents
            x_min, x_max = float(nodes[:, 0].min()), float(nodes[:, 0].max())
            y_min, y_max = float(nodes[:, 1].min()), float(nodes[:, 1].max())
            dx, dy = x_max - x_min, y_max - y_min
            ax.set_xlim(x_min - 0.05 * dx, x_max + 0.05 * dx)
            ax.set_ylim(y_min - 0.05 * dy, y_max + 0.05 * dy)

            # release heavy state object before moving to next subplot
            del state
            import gc

            gc.collect()

    sm = plt.cm.ScalarMappable(cmap="plasma", norm=plt.Normalize(vmin=field_min, vmax=field_max))
    sm.set_array([])
    # leave space at the bottom for a horizontal colorbar
    fig.subplots_adjust(bottom=0.15, hspace=0.3, wspace=0.25)
    fig.colorbar(
        sm,
        ax=axes.ravel().tolist(),
        orientation="horizontal",
        label="temperature",
        fraction=0.04,
        pad=0.08,
    )

    if config.save:
        fig.savefig(
            Path(config.outputs_path) / "tarzia_problem_grid.png",
            bbox_inches="tight",
            dpi=300,
        )
    if config.show:
        plt.show()
    plt.close(fig)


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




if __name__ == "__main__":
    main(Config(outputs_path="./output/BOT2023", force=False, save=True, show=False).init())
