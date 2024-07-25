"""
Created at 21.08.2019
"""

from dataclasses import dataclass

from matplotlib import pyplot as plt

from conmech.helpers.config import Config
from conmech.mesh.boundaries_description import BoundariesDescription
from conmech.plotting.drawer import Drawer
from conmech.scenarios.problems import StaticDisplacementProblem
from conmech.simulations.problem_solver import StaticSolver
from conmech.properties.mesh_description import (
    RectangleMeshDescription,
    CubeMeshDescription,
)

from conmech.dynamics.contact.relu_slope_contact_law import make_slope_contact_law


@dataclass
class StaticSetup(StaticDisplacementProblem):
    mu_coef: ... = 4
    la_coef: ... = 4
    contact_law: ... = make_slope_contact_law(slope=1)

    @staticmethod
    def inner_forces(x, t=None):
        return -0.2 * x

    @staticmethod
    def outer_forces(x, t=None):
        return 0 * x

    boundaries: ... = BoundariesDescription(
        contact=lambda x: x[1] == 0, dirichlet=lambda x: x[0] == 0
    )


def main(config: Config, dimension=2):
    """
    Entrypoint to example.

    To see result of simulation you need to call from python `main(Config().init())`.
    """
    solving_method = "schur" if dimension == 2 else "global"

    if dimension == 2:
        mesh_descr = RectangleMeshDescription(
            initial_position=None, max_element_perimeter=0.5, scale=[2.5, 1]
        )
    else:
        mesh_descr = CubeMeshDescription(initial_position=None)

    setup = StaticSetup(mesh_descr=mesh_descr)
    runner = StaticSolver(setup, solving_method)

    state = runner.solve(verbose=True, initial_displacement=setup.initial_displacement)
    if dimension == 2:
        Drawer(state=state, config=config).draw(show=config.show, save=config.save)
    else:
        fig = plt.figure()
        axs = fig.add_subplot(111, projection="3d")
        # Draw nodes
        nodes = state.body.mesh.nodes
        axs.scatter(nodes[:, 0], nodes[:, 1], nodes[:, 2], c="b", marker="o")

        # Draw elements
        faces = state.displaced_nodes[state.body.mesh.boundary_surfaces]
        from mpl_toolkits.mplot3d.art3d import Poly3DCollection

        axs.add_collection3d(
            Poly3DCollection(faces, facecolors="cyan", linewidths=1, edgecolors="r", alpha=0.25)
        )
        plt.show()


if __name__ == "__main__":
    main(Config().init(), 2)
