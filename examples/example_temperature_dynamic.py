"""
Created at 21.08.2019
"""
from dataclasses import dataclass

import numpy as np

from conmech.problem_solver import TDynamic as TDynamicProblemSolver
from conmech.problems import Dynamic
from examples.p_slope_contact_law import make_slope_contact_law
from utils.drawer import Drawer


class TPSlopeContactLaw(make_slope_contact_law(slope=1)):
    def g(self, t):
        return 10.7 + t * 0.02
        # return 0.5 + t * 0.01

    def hnu(self, uN, t):
        if uN > self.g(t):
            return 100. * (uN - self.g(t))
        return 0

    def htau(self, uN, x):
        # return 0
        return 0.1 * self.hnu(uN, x)

    def jT(self, vTx, vTy):
        # return np.log(np.linalg.norm(vTx, vTy)+1)
        return np.linalg.norm(vTx, vTy)

    @staticmethod
    def h_temp(vTnorm):
        return 0.  # 0.1*vTnorm


@dataclass()
class TDynamicSetup(Dynamic):
    grid_height: ... = 1
    cells_number: ... = (2, 5)
    inner_forces: ... = np.array([0.0, 1.5])
    outer_forces: ... = np.array([0., 0])
    mu_coef: ... = 4
    lambda_coef: ... = 4
    th_coef: ... = 2
    ze_coef: ... = 2
    time_step: ... = 0.1
    contact_law: ... = TPSlopeContactLaw

    @staticmethod
    def friction_bound(u_nu):
        return 0


if __name__ == '__main__':
    setup = TDynamicSetup()
    runner = TDynamicProblemSolver(setup, solving_method='schur')

    states = runner.solve(n_steps=100, output_step=(0, 50, 90, 130), verbose=True)
    T_max = 0
    for state in states:
        T_max = np.max(state.temperature)
    for state in states:
        Drawer(state).draw(temp_max=T_max)
