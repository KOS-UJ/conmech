"""
Created at 21.08.2019
"""
from dataclasses import dataclass

import numpy as np
from conmech.helpers.config import Config
from conmech.mesh.boundaries_description import BoundariesDescription
from conmech.plotting.drawer import Drawer
from conmech.scenarios.problems import ContactLaw, StaticDisplacementProblem
from conmech.simulations.problem_solver import StaticSolver as StaticProblemSolver
from conmech.mesh.mesh import MeshProperties

class JureczkaOchal2019(ContactLaw):
    @staticmethod
    def potential_normal_direction(u_nu: float) -> float:
        if u_nu <= 0:
            return 0.0
        if u_nu < 0.1:
            return 10 * u_nu * u_nu
        return 0.1

    @staticmethod
    def potential_tangential_direction(u_tau: np.ndarray) -> float:
        return np.log(np.sum(u_tau * u_tau) ** 0.5 + 1)

    @staticmethod
    def subderivative_normal_direction(u_nu: float, v_nu: float) -> float:
        return 0

    @staticmethod
    def regularized_subderivative_tangential_direction(
        u_tau: np.ndarray, v_tau: np.ndarray, rho=1e-7
    ) -> float:
        """
        Coulomb regularization
        """
        return 0
        # regularization = 1 / np.sqrt(u_tau[0] * u_tau[0] + u_tau[1] * u_tau[1] + rho ** 2)
        # result = regularization * (u_tau[0] * v_tau[0] + u_tau[1] * v_tau[1])
        # return result


@dataclass
class StaticSetup(StaticDisplacementProblem):
    mu_coef: ... = 4
    la_coef: ... = 4
    contact_law: ... = JureczkaOchal2019

    @staticmethod
    def inner_forces(x, t=None):
        return np.array([-1.2, -0.8])

    @staticmethod
    def outer_forces(x, t=None):
        return np.array([0, 0])

    @staticmethod
    def friction_bound(u_nu: float) -> float:
        if u_nu < 0:
            return 0
        if u_nu < 0.1:
            return 8 * u_nu
        return 0.8

    boundaries: ... = BoundariesDescription(
        contact=lambda x: x[1] == 0, dirichlet=lambda x: x[0] == 0
    )


def main(config: Config):
    """
    Entrypoint to example.

    To see result of simulation you need to call from python `main(Config().init())`.
    """
    mesh_prop = MeshProperties(
        mesh_type="cross",
        mesh_density=[16, 8],
        grid_height=1
    )
    if config.test:
        mesh_prop.mesh_density = [4, 2]

    setup = StaticSetup(mesh_prop)
    runner = StaticProblemSolver(setup, "schur")

    state = runner.solve(
        verbose=True,
        fixed_point_abs_tol=0.001,
        initial_displacement=setup.initial_displacement,
    )
    Drawer(state=state, config=config).draw(show=config.show, save=config.save)


if __name__ == "__main__":
    main(Config().init())
