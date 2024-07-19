"""
Created at 21.08.2019
"""

from dataclasses import dataclass

import numpy as np
from conmech.helpers.config import Config
from conmech.mesh.boundaries_description import BoundariesDescription
from conmech.plotting.drawer import Drawer
from conmech.scenarios.problems import QuasistaticDisplacementProblem
from conmech.simulations.problem_solver import TimeDependentSolver
from conmech.properties.mesh_description import CrossMeshDescription

from conmech.dynamics.contact.relu_slope_contact_law import make_slope_contact_law


@dataclass()
class QuasistaticSetup(QuasistaticDisplacementProblem):
    mu_coef: ... = 4
    la_coef: ... = 4
    th_coef: ... = 4
    ze_coef: ... = 4
    time_step: ... = 0.1
    contact_law: ... = make_slope_contact_law(slope=1e1)

    @staticmethod
    def inner_forces(x, t=None):
        return np.array([-0.2, -0.2])

    @staticmethod
    def outer_forces(x, t=None):
        return np.array([0, 0])

    @staticmethod
    def friction_bound(u_nu: float) -> float:
        return 0

    boundaries: ... = BoundariesDescription(
        contact=lambda x: x[1] == 0, dirichlet=lambda x: x[0] == 0
    )


def main(config: Config):
    """
    Entrypoint to example.

    To see result of simulation you need to call from python `main(Config().init())`.
    """
    mesh_descr = CrossMeshDescription(
        initial_position=None, max_element_perimeter=0.5, scale=[2.5, 1]
    )
    setup = QuasistaticSetup(mesh_descr)
    runner = TimeDependentSolver(setup, solving_method="schur")

    states = runner.solve(
        n_steps=32,
        output_step=(0, 32),
        verbose=True,
        initial_displacement=setup.initial_displacement,
        initial_velocity=setup.initial_velocity,
    )
    for state in states:
        Drawer(state=state, config=config).draw(show=config.show, save=config.save)


if __name__ == "__main__":
    main(Config().init())
