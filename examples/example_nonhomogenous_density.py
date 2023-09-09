"""
Created at 02.09.2023
"""
from dataclasses import dataclass

import numpy as np
from conmech.helpers.config import Config
from conmech.mesh.boundaries_description import BoundariesDescription
from conmech.plotting.drawer import Drawer
from conmech.scenarios.problems import StaticDisplacementProblem
from conmech.simulations.problem_solver import NonHomogenousSolver
from conmech.properties.mesh_properties import CrossMeshDescription

from examples.p_slope_contact_law import make_slope_contact_law


@dataclass
class StaticSetup(StaticDisplacementProblem):
    mu_coef: ... = 4
    la_coef: ... = 4
    contact_law: ... = make_slope_contact_law(slope=1)

    @staticmethod
    def inner_forces(x, t=None):
        return np.array([0, 0])

    @staticmethod
    def outer_forces(x, t=None):
        return np.array([0, 0.1]) if x[1] < 0.2 and x[0] > 2.2 else np.array([0, 0])

    @staticmethod
    def friction_bound(u_nu: float) -> float:
        return 0

    boundaries: ... = BoundariesDescription(
        contact=lambda x: x[1] == 0 and x[0] < 0.2, dirichlet=lambda x: x[0] == 0
    )


def main(config: Config):
    """
    Entrypoint to example.

    To see result of simulation you need to call from python `main(Config().init())`.
    """
    mesh_descr = CrossMeshDescription(
        initial_position=None,
        max_element_perimeter=0.5,
        scale=[2.5, 1]
    )
    setup = StaticSetup(mesh_descr)
    runner = NonHomogenousSolver(setup, "schur")

    elem_centers = np.empty(shape=(len(runner.body.mesh.elements), 2))
    for idx, elem in enumerate(runner.body.mesh.elements):
        verts = runner.body.mesh.initial_nodes[elem]
        elem_centers[idx] = np.sum(verts, axis=0) / len(elem)
    density = np.asarray([1 if x[0] < 1 else 0.2 for x in elem_centers])

    runner.update_density(density)

    state = runner.solve(verbose=True, initial_displacement=setup.initial_displacement)
    Drawer(state=state, config=config).draw(show=config.show, save=config.save)


if __name__ == "__main__":
    main(Config().init())
