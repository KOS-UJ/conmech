"""
Created at 21.08.2019
"""

from typing import Callable
import numpy as np
from deep_conmech.common import basic_helpers

from conmech.vertex_utils import length


class F:
    def __init__(self, mesh, inter_forces: Callable, outer_forces: Callable):
        self.inter_forces = inter_forces
        self.outer_forces = outer_forces
        self.mesh = mesh
        self.F = np.zeros([self.mesh.independent_nodes_count, 2])

    @property
    def F_vector(self):
        return basic_helpers.stack_column(self.F).reshape(-1)


    def setF(self):
        # f0 = np.array([self.f0(p) for p in self.mesh.moved_points])
        # F = self.mesh.AREA @ f0

        F = np.zeros([self.mesh.nodes_count, 2])

        for element_id, element in enumerate(self.mesh.cells):
            p0 = self.mesh.initial_points[element[0]]
            p1 = self.mesh.initial_points[element[1]]
            p2 = self.mesh.initial_points[element[2]]

            f0 = self.inter_forces(*p0)
            f1 = self.inter_forces(*p1)
            f2 = self.inter_forces(*p2)

            f_mean = (f0 + f1 + f2) / 3  # TODO

            F[element[0]] += (
                f_mean / 3 * self.mesh.element_initial_area[element_id]
            )
            F[element[1]] += (
                f_mean / 3 * self.mesh.element_initial_area[element_id]
            )
            F[element[2]] += (
                f_mean / 3 * self.mesh.element_initial_area[element_id]
            )

        # np.allclose(self.F2,  self.F)
        for neumann_boundary in self.mesh.boundaries.neumann:

            for i in range(1, len(neumann_boundary)):
                v0 = neumann_boundary[i - 1]
                v1 = neumann_boundary[i]

                edge_length = length(
                    self.mesh.initial_points[v0], self.mesh.initial_points[v1]
                )
                v_mid = (
                    self.mesh.initial_points[v0] + self.mesh.initial_points[v1]
                ) / 2

                f_neumann = self.outer_forces(*v_mid) * edge_length / 2

                F[v0] += f_neumann
                F[v1] += f_neumann

        self.F = F[:self.mesh.independent_nodes_count, :]
