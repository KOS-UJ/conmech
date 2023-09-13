"""
Created at 05.09.2023
"""
from dataclasses import dataclass

import numpy as np
from conmech.helpers.config import Config
from conmech.mesh.boundaries_description import BoundariesDescription
from conmech.plotting.drawer import Drawer
from conmech.scenarios.problems import StaticDisplacementProblem
from conmech.simulations.problem_solver import StaticSolver
from conmech.properties.mesh_properties import ImportedMeshDescription


from examples.p_slope_contact_law import make_slope_contact_law


E = 10000
kappa = 0.4


@dataclass
class StaticSetup(StaticDisplacementProblem):
    mu_coef: ... = E / (1 + kappa)
    la_coef: ... = E * kappa / ((1 + kappa) * (1 - 2 * kappa))
    contact_law: ... = make_slope_contact_law(slope=1)

    @staticmethod
    def inner_forces(x, t=None):
        return np.array([0, 0])

    @staticmethod
    def outer_forces(x, t=None):
        return np.array([0, -1]) if x[0] > 1.9 and x[1] < 0.1 else np.zeros(2)

    @staticmethod
    def friction_bound(u_nu: float) -> float:
        return 0

    boundaries: ... = BoundariesDescription(
        contact=lambda x: x[1] == 0 and x[0] < 0.5, dirichlet=lambda x: x[0] == 0
    )


def main(config: Config):
    """
    Entrypoint to example.

    To see result of simulation you need to call from python `main(Config().init())`.
    """
    mesh_descr = ImportedMeshDescription(
        initial_position=None, path="examples/meshes/example_mesh.msh"
    )
    setup = StaticSetup(mesh_descr)
    runner = StaticSolver(setup, "schur")

    state = runner.solve(verbose=True, initial_displacement=setup.initial_displacement)
    Drawer(state=state, config=config).draw(show=config.show, save=config.save)


if __name__ == "__main__":
    main(Config().init())
