"""
Created at 22.02.2021
"""

import numpy as np
from typing import Tuple

from simulation.solvers.optimization.optimization import Optimization


class SchurComplement(Optimization):

    def __init__(self, grid, inner_forces, outer_forces, mu_coef, lambda_coef, contact_law, friction_bound):
        super().__init__(grid, inner_forces, outer_forces, mu_coef, lambda_coef, contact_law, friction_bound)

        contact_points_num = grid.contact_num
        independent_points_num = grid.independent_num()
        contact_ids = slice(0, contact_points_num)
        free_ids = slice(contact_points_num, independent_points_num)

        free_x_free = SchurComplement.get_submatrix(self.B, indices=(free_ids, free_ids))
        free_x_contact = SchurComplement.get_submatrix(self.B, indices=(free_ids, contact_ids))
        contact_x_free = SchurComplement.get_submatrix(self.B, indices=(contact_ids, free_ids))
        contact_x_contact = SchurComplement.get_submatrix(self.B, indices=(contact_ids, contact_ids))

        self.free_x_contact = free_x_contact
        self.free_x_free_inverted = np.linalg.inv(free_x_free)
        _point_relations = np.dot(self.free_x_free_inverted, self.free_x_contact)
        _point_relations = np.dot(contact_x_free, _point_relations)
        _point_relations = contact_x_contact - _point_relations
        self.__point_relations = np.asarray(_point_relations)

        forces_contact = np.append(self.forces.Zero[contact_ids], self.forces.One[contact_ids]).reshape(-1, 1)
        self.forces_free = np.append(self.forces.Zero[free_ids], self.forces.One[free_ids]).reshape(-1, 1)
        point_forces = np.dot(self.free_x_free_inverted, self.forces_free)
        point_forces = np.dot(contact_x_free, point_forces)
        point_forces = forces_contact - point_forces
        self.__point_forces = np.asarray(point_forces.reshape(1, -1))

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

        c_num = self.grid.BorderEdgesC

        x = initial_guess.reshape(2,-1)
        y = x[:, 0: c_num]
        z = y.reshape(1,-1)
        ut_vector = z

        ut_vector = super().solve(ut_vector)

        ut_v = ut_vector.reshape(-1,1)
        first = np.dot(self.free_x_contact, ut_v)
        second = self.forces_free - first
        ui_vector = np.dot(self.free_x_free_inverted, second)

        ut = ut_vector.reshape(2, -1)
        ui = ui_vector.reshape(2, -1)
        result = np.concatenate((ut, ui), axis=1)
        result = result.reshape(1, -1)
        result = np.squeeze(np.asarray(result))

        return result
