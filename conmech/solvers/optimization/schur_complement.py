"""
Created at 22.02.2021
"""

import numpy as np
from typing import Tuple

from conmech.solvers.optimization.optimization import Optimization


class SchurComplement(Optimization):

    def __init__(self, grid, inner_forces, outer_forces, mu_coef,
                 lambda_coef, th_coef, ze_coef, time_step, contact_law, friction_bound):
        super().__init__(grid, inner_forces, outer_forces, mu_coef,
                         lambda_coef, th_coef, ze_coef, time_step, contact_law, friction_bound)

        contact_ids = slice(0, grid.contact_num)
        free_ids = slice(grid.contact_num, grid.independent_num)

        #free_x_free = SchurComplement.get_submatrix(self.B, indices=(free_ids, free_ids))
        #free_x_contact = SchurComplement.get_submatrix(self.B, indices=(free_ids, contact_ids))
        #contact_x_free = SchurComplement.get_submatrix(self.B, indices=(contact_ids, free_ids))
        #contact_x_contact = SchurComplement.get_submatrix(self.B, indices=(contact_ids, contact_ids))

        free_x_free = SchurComplement.get_submatrix(self.A, indices=(free_ids, free_ids))
        free_x_contact = SchurComplement.get_submatrix(self.A, indices=(free_ids, contact_ids))
        contact_x_free = SchurComplement.get_submatrix(self.A, indices=(contact_ids, free_ids))
        contact_x_contact = SchurComplement.get_submatrix(self.A, indices=(contact_ids, contact_ids))

        self.free_x_contact = free_x_contact
        self.free_x_free_inverted = np.linalg.inv(free_x_free)
        _point_relations = np.dot(self.free_x_free_inverted, self.free_x_contact)
        _point_relations = np.dot(contact_x_free, _point_relations)
        _point_relations = contact_x_contact - _point_relations
        self.__point_relations = np.asarray(_point_relations)

        #ADDED When working with velocity v, forces_contact depend on u
        time_dependent = 1
        if time_dependent:
            #X = np.squeeze(np.asarray(np.dot(self.B, scipy.sparse.lil_matrix(self.uVector).transpose()).todense()))
            Big_B = np.bmat([[self.B[0, 0], self.B[0, 1]],
                          [self.B[1, 0], self.B[1, 1]]])

            X = np.dot(Big_B, self.u_vector.T)
            X2 = X.reshape((2, -1))
            X_Zero = np.asarray(X2)[0]
            X_One = np.asarray(X2)[1]
            #Ebig = self.FVector - X
            #Et = np.append(Ebig[self.i:self.n], Ebig[self.n + self.i:self.n + self.n])
            forces_contact = np.append(self.forces.Zero[contact_ids] - X_Zero[contact_ids]
                                       , self.forces.One[contact_ids] - X_One[contact_ids]).reshape(-1, 1)
            #self.Ei = np.append(Ebig[0:self.i], Ebig[self.n:self.n + self.i])
            self.forces_free = np.append(self.forces.Zero[free_ids] - X_Zero[free_ids],
                                         self.forces.One[free_ids] - X_One[free_ids]).reshape(-1, 1)
            #CiiINVEi = multiplyByDAT('E:\\SPARE\\cross ' + str(self.SizeH) + ' CiiINV.dat', self.Ei)
            point_forces = np.dot(self.free_x_free_inverted, self.forces_free)
            point_forces = np.dot(contact_x_free, point_forces)
            #self.E = (Et - np.asarray(self.Cti.dot(CiiINVEi))).astype(np.single)
            point_forces = forces_contact - point_forces
            self.__point_forces = np.asarray(point_forces.reshape(1, -1))
        else:
            #Ebig = self.FVector
            #Et = np.append(Ebig[self.i:self.n], Ebig[self.n + self.i:self.n + self.n])
            forces_contact = np.append(self.forces.Zero[contact_ids], self.forces.One[contact_ids]).reshape(-1, 1)
            #self.Ei = np.append(Ebig[0:self.i], Ebig[self.n:self.n + self.i])
            self.forces_free = np.append(self.forces.Zero[free_ids], self.forces.One[free_ids]).reshape(-1, 1)
            # CiiINVEi = multiplyByDAT(self.disc + '\\cross ' + str(self.SizeH) + ' CiiINV.dat', self.Ei)
            point_forces = np.dot(self.free_x_free_inverted, self.forces_free)
            point_forces = np.dot(contact_x_free, point_forces)
            #self.E = (Et - np.asarray(self.Cti.dot(CiiINVEi))).astype(np.single)
            point_forces = forces_contact - point_forces
            self.__point_forces = np.asarray(point_forces.reshape(1, -1))

    def __repr__(self):
        return "schur"

    @staticmethod
    def get_submatrix(arrays: iter, indices: Tuple[slice, slice]) -> np.matrix:
        result = np.bmat([[arrays[0, 0][indices], arrays[0, 1][indices]],
                          [arrays[1, 0][indices], arrays[1, 1][indices]]])
        return result

    @property
    def point_relations(self) -> np.ndarray:
        return self.__point_relations

    @property
    def point_forces(self) -> np.ndarray:
        return self.__point_forces

    def solve(self, initial_guess: np.ndarray) -> np.ndarray:
        truncated_initial_guess = self.truncate_free_points(initial_guess)
        solution_contact = super().solve(truncated_initial_guess)
        solution_free = self.complement_free_points(solution_contact)
        solution = self.merge(solution_contact, solution_free)
        return solution

    def truncate_free_points(self, initial_guess: np.ndarray) -> np.ndarray:
        _result = initial_guess.reshape(2, -1)
        _result = _result[:, 0: self.grid.contact_num]
        _result = _result.reshape(1, -1)
        result = _result
        return result

    def complement_free_points(self, truncated_solution: np.ndarray) -> np.ndarray:
        _result = truncated_solution.reshape(-1, 1)
        _result = np.dot(self.free_x_contact, _result)
        _result = self.forces_free - _result
        result = np.dot(self.free_x_free_inverted, _result)
        return result

    @staticmethod
    def merge(solution_contact: np.ndarray, solution_free: np.ndarray) -> np.ndarray:
        u_contact = solution_contact.reshape(2, -1)
        u_free = solution_free.reshape(2, -1)
        _result = np.concatenate((u_contact, u_free), axis=1)
        _result = _result.reshape(1, -1)
        result = np.squeeze(np.asarray(_result))
        return result
