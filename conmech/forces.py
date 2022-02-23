"""
Created at 21.08.2019
"""
from typing import Callable

import numpy as np
from conmech.vertex_utils import length


class Forces:
    def __init__(self, mesh, inter_forces: Callable, outer_forces: Callable):
        self.inter_forces = inter_forces
        self.outer_forces = outer_forces
        self.mesh = mesh
        self.F = np.zeros([self.mesh.independent_num, 2])
        self.Zero = np.zeros([self.mesh.independent_num])
        self.One = np.zeros([self.mesh.independent_num])

    def setF(self):
        self.F = np.zeros([self.mesh.points_number, 2])

        for element_id, element in enumerate(self.mesh.cells):
            p0 = self.mesh.initial_points[element[0]]
            p1 = self.mesh.initial_points[element[1]]
            p2 = self.mesh.initial_points[element[2]]

            f0 = self.inter_forces(*p0)
            f1 = self.inter_forces(*p1)
            f2 = self.inter_forces(*p2)

            f_mean = (f0 + f1 + f2) / 3

            self.F[element[0]] += f_mean / 3 * self.mesh.element_area[element_id]
            self.F[element[1]] += f_mean / 3 * self.mesh.element_area[element_id]
            self.F[element[2]] += f_mean / 3 * self.mesh.element_area[element_id]

        for neumann_boundary in self.mesh.boundaries.neumann:

            for i in range(1, len(neumann_boundary)):
                v0 = neumann_boundary[i - 1]
                v1 = neumann_boundary[i]

                edge_length = length(self.mesh.initial_points[v0], self.mesh.initial_points[v1])
                v_mid = (self.mesh.initial_points[v0] + self.mesh.initial_points[v1]) / 2

                f_neumann = self.outer_forces(*v_mid) * edge_length / 2

                self.F[v0] += f_neumann
                self.F[v1] += f_neumann

        self.Zero = self.F[:, 0]
        self.One = self.F[:, 1]
