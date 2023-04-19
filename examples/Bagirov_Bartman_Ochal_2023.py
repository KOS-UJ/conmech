"""
Created at 21.08.2019
"""
from dataclasses import dataclass

import numpy as np
from conmech.helpers.config import Config
from conmech.mesh.boundaries_description import BoundariesDescription
from conmech.plotting.drawer import Drawer
from conmech.scenarios.problems import ContactLaw, Static
from conmech.simulations.problem_solver import Static as StaticProblemSolver


class JureczkaOchal2019(ContactLaw):
    @staticmethod
    def potential_normal_direction(u_nu: float) -> float:
        # if u_nu <= 0:
        #     return 0.0
        # if u_nu < 0.1:
        #     return 10 * u_nu * u_nu
        # return 0.1
        if u_nu >= 0:
            return 0.0
        if u_nu > -0.5e-3:
            return 30e9 * u_nu ** 2
        if u_nu > -1e-3:
            return 10e9 * (u_nu ** 2 + u_nu)
        if u_nu > -2e-3:
            return 5e9 * (u_nu ** 2 + u_nu)
        return 0.0

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


mesh_density = 4


@dataclass()
class StaticSetup(Static):
    grid_height: ... = 0.1
    elements_number: ... = (mesh_density, 8 * mesh_density)
    mu_coef: ... = 5.3e10
    la_coef: ... = 7.95e10
    contact_law: ... = JureczkaOchal2019

    @staticmethod
    def inner_forces(x, t=None):
        return np.array([0.0, 0.0])

    @staticmethod
    def outer_forces(x, t=None):
        return np.array([0, 5e6])

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


def main(show: bool = True, save: bool = False):
    setup = StaticSetup(mesh_type="cross")

    for force in np.arange(20e6, 30e6 + 1, 2e6):
        def outer_forces(x, t=None):
            return np.array([0, force])

        setup.outer_forces = outer_forces

        runner = StaticProblemSolver(setup, "schur")

        state = runner.solve(
            verbose=True,
            fixed_point_abs_tol=0.001,
            initial_displacement=setup.initial_displacement,
            method="BFGS"
        )
        config = Config()
        drawer = Drawer(state=state, config=config)
        drawer.colorful = True
        drawer.draw(show=show, save=save)


if __name__ == "__main__":
    main(show=True, save=False)
