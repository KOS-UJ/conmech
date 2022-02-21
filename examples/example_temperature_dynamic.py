"""
Created at 21.08.2019
"""
from dataclasses import dataclass

import numpy as np

from conmech.problem_solver import TDynamic as TDynamicProblemSolver
from conmech.problems import Dynamic
from examples.p_slope_contact_law import make_slope_contact_law
from utils.drawer import Drawer


class TPSlopeContactLaw(make_slope_contact_law(slope=1e1)):
    # @staticmethod
    # def g(t):
    #     return 10.7 + t * 0.02
        # return 0.5 + t * 0.01

    @staticmethod
    def h_nu(uN, t):
        g_t = 10.7 + t * 0.02
        if uN > g_t:
            return 100. * (uN - g_t)
        return 0


    @staticmethod
    def h_tau(uN, t):
        # return 0
        g_t = 10.7 + t * 0.02
        if uN > g_t:
            return 10. * (uN - g_t)
        return 0
    #
    # def jT(self, vTx, vTy):
    #     # return np.log(np.linalg.norm(vTx, vTy)+1)
    #     return np.linalg.norm(vTx, vTy)

    @staticmethod
    def h_temp(vTnorm):
        return 0.1 * vTnorm


@dataclass()
class TDynamicSetup(Dynamic):
    grid_height: ... = 1.
    cells_number: ... = (10, 25)
    mu_coef: ... = 4
    lambda_coef: ... = 4
    th_coef: ... = 4
    ze_coef: ... = 4
    time_step: ... = 0.02
    contact_law: ... = TPSlopeContactLaw

    @staticmethod
    def inner_forces(x, y):
        return np.array([0., -1.])

    @staticmethod
    def outer_forces(x, y):
        return np.array([0, 0])

    @staticmethod
    def friction_bound(u_nu):
        return 0

    @staticmethod
    def is_contact(x, y):
        return y == 0

    @staticmethod
    def is_dirichlet(x, y):
        return x == 0


if __name__ == '__main__':
    setup = TDynamicSetup()
    runner = TDynamicProblemSolver(setup, solving_method='schur')

    states = runner.solve(n_steps=32, output_step=range(0, 32, 4), verbose=True)
    T_max = 0
    for state in states:
        T_max = np.max(state.temperature)
    for state in states:
        Drawer(state).draw(temp_max=T_max)
