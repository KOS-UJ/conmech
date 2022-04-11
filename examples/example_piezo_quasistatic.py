"""
Created at 21.08.2019
"""
from dataclasses import dataclass

import numpy as np

from conmech.problem_solver import TDynamic as TDynamicProblemSolver, PQuasistatic
from conmech.problems import Dynamic, Quasistatic
from conmech.utils.drawer import Drawer
from examples.p_slope_contact_law import make_slope_contact_law


class TPSlopeContactLaw(make_slope_contact_law(slope=1e1)):
    # @staticmethod  # TODO # 48
    # def g(t):
    #     return 10.7 + t * 0.02
    # return 0.5 + t * 0.01

    @staticmethod
    def h_nu(uN, t):
        g_t = 10.7 + t * 0.02
        if uN > g_t:
            return 100.0 * (uN - g_t)
        return 0

    @staticmethod
    def h_tau(uN, t):
        g_t = 10.7 + t * 0.02
        if uN > g_t:
            return 10.0 * (uN - g_t)
        return 0

    # def jT(self, vTx, vTy):  # TODO # 48
    #     # return np.log(np.linalg.norm(vTx, vTy)+1)
    #     return np.linalg.norm(vTx, vTy)

    @staticmethod
    def h_temp(u_tau):  # potential  # TODO # 48
        return 0.1 * 0.5 * u_tau ** 2


@dataclass()
class PQuasistaticSetup(Quasistatic):
    grid_height: ... = 1.0
    elements_number: ... = (4, 10)
    mu_coef: ... = 4
    la_coef: ... = 4
    th_coef: ... = 4
    ze_coef: ... = 4
    time_step: ... = 0.02
    contact_law: ... = TPSlopeContactLaw

    @staticmethod
    def initial_electric_potential(x: np.ndarray) -> np.ndarray:
        return np.asarray([0.])

    @staticmethod
    def inner_forces(x, y):
        return np.array([0.0, -1.0])

    @staticmethod
    def outer_forces(x, y):
        if x == 0:
            return np.array([48. * (0.25 - (y - .5) ** 2), 0])
        if x == 2.5:
            return np.array([-48. * (0.25 - (y - .5) ** 2), 0])
        return np.array([0, 0])

    @staticmethod
    def friction_bound(u_nu):
        return 0

    @staticmethod  # TODO #49
    def is_contact(x):
        return x[1] == 0

    @staticmethod  # TODO #49
    def is_dirichlet(x):
        return x[0] == 0


def main(show: bool):
    setup = PQuasistaticSetup()
    runner = PQuasistatic(setup, solving_method="piezo")

    states = runner.solve(n_steps=32, output_step=range(0, 32, 4), verbose=True,
                          initial_displacement=setup.initial_displacement,
                          initial_velocity=setup.initial_velocity,
                          initial_electric_potential=setup.initial_electric_potential)
    e_max = -np.inf
    e_min = np.inf
    for state in states:
        e_max = max(e_max, np.max(state.electric_potential))
        e_min = min(e_min, np.min(state.electric_potential))
    for state in states:
        Drawer(state).draw(temp_max=e_max, temp_min=e_min, show=show)


if __name__ == "__main__":
    main(show=True)
