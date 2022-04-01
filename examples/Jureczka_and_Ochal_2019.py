"""
Created at 21.08.2019
"""
from dataclasses import dataclass

import numpy as np

from conmech.problem_solver import Static as StaticProblemSolver
from conmech.problems import ContactLaw
from conmech.problems import Static
from conmech.utils.drawer import Drawer


class JureczkaOchal2018(ContactLaw):
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


@dataclass()
class StaticSetup(Static):
    grid_height: ... = 1.0
    elements_number: ... = (4, 8)
    mu_coef: ... = 4
    la_coef: ... = 4
    contact_law: ... = JureczkaOchal2018

    @staticmethod
    def inner_forces(x, y):
        return np.array([-1.2, -0.8])

    @staticmethod
    def outer_forces(x, y):
        return np.array([0, 0])

    @staticmethod
    def friction_bound(u_nu: float) -> float:
        if u_nu < 0:
            return 0
        if u_nu < 0.1:
            return 8 * u_nu
        return 0.8

    @staticmethod
    def is_contact(x):
        return x[1] == 0

    @staticmethod
    def is_dirichlet(x):
        return x[0] == 0


def main(show: bool):
    setup = StaticSetup()
    runner = StaticProblemSolver(setup, "schur")

    state = runner.solve(verbose=True, fixed_point_abs_tol=0.001,
                         initial_displacement=setup.initial_displacement)
    Drawer(state).draw(show=show)


if __name__ == "__main__":
    main(show=True)
