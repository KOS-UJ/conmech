"""
Created at 22.02.2021
"""

import numpy as np
from simulation.solvers.optimization.optimization import Optimization


class SchurComplement(Optimization):

    def __init__(self, grid, inner_forces, outer_forces, mu_coef, lambda_coef, contact_law, friction_bound):
        super().__init__(grid, inner_forces, outer_forces, mu_coef, lambda_coef, contact_law, friction_bound)

        C11 = self.B[0, 0]
        C12 = self.B[0, 1]
        C21 = self.B[1, 0]
        C22 = self.B[1, 1]

        c_num = grid.BorderEdgesC
        i_num = grid.indNumber()

        indices = (slice(c_num, i_num), slice(c_num, i_num))
        self.Cii = np.bmat([[C11[indices], C12[indices]],
                             [C21[indices], C22[indices]]])

        indices = (slice(c_num, i_num), slice(0, c_num))
        self.Cit = np.bmat([[C11[indices], C12[indices]],
                             [C21[indices], C22[indices]]])

        indices = (slice(0, c_num), slice(c_num, i_num))
        self.Cti = np.bmat([[C11[indices], C12[indices]],
                             [C21[indices], C22[indices]]])

        indices = (slice(0, c_num), slice(0, c_num))
        self.Ctt = np.bmat([[C11[indices], C12[indices]],
                             [C21[indices], C22[indices]]])

        self.CiiINV = np.linalg.inv(self.Cii)
        self.CiiINVCit = np.dot(self.CiiINV, self.Cit)
        self.CtiCiiINVCit = np.dot(self.Cti, self.CiiINVCit)
        self._C = self.Ctt - self.CtiCiiINVCit
        self._C = np.asarray(self.C)

        indices = slice(c_num, i_num)
        self.Ebig = np.append(self.forces.Zero, self.forces.One).reshape(-1,1)
        self.Ei = np.append(self.forces.Zero[indices], self.forces.One[indices]).reshape(-1,1)

        indices = slice(0, c_num)
        self.Et = np.append(self.forces.Zero[indices], self.forces.One[indices]).reshape(-1,1)

        self.CiiINVEi = np.dot(self.CiiINV, self.Ei)
        self.CtiCiiINVEi = np.dot(self.Cti, self.CiiINVEi)

        self._E = self.Et - self.CtiCiiINVEi
        self._E = np.asarray(self.E.reshape(1, -1))

    @property
    def C(self):
        return self._C

    @property
    def E(self):
        return self._E

    def solve(self, initial_guess: np.ndarray) -> np.ndarray:

        c_num = self.grid.BorderEdgesC

        x = initial_guess.reshape(2,-1)
        y = x[:, 0: c_num]
        z = y.reshape(1,-1)
        ut_vector = z

        ut_vector = super().solve(ut_vector)

        ut_v = ut_vector.reshape(-1,1)
        first = np.dot(self.Cit, ut_v)
        second = self.Ei - first
        ui_vector = np.dot(self.CiiINV, second)

        ut = ut_vector.reshape(2, -1)
        ui = ui_vector.reshape(2, -1)
        result = np.concatenate((ut, ui), axis=1)
        result = result.reshape(1, -1)
        result = np.squeeze(np.asarray(result))

        return result
