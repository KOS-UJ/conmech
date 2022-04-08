"""
Created at 21.08.2019
"""

from typing import Callable

import numpy as np

from conmech.helpers import nph


class Forces:
    """
    Represents body forces at each node: inner and outer.
    """

    def __init__(self, mesh, inter_forces: Callable, outer_forces: Callable):
        self.inter_forces = inter_forces
        self.outer_forces = outer_forces
        self.mesh = mesh

        self._forces = np.zeros([self.mesh.independent_nodes_count, 2])

    @property
    def forces(self):
        return self._forces[: self.mesh.independent_nodes_count, :]

    @property
    def forces_vector(self):
        return nph.stack_column(self.forces).reshape(-1)

    def update_forces(self):
        """
        Recalculate `forces`
        """
        # f_0 = np.array([self.f_0(p) for p in self.mesh.moved_nodes])
        # F = self.mesh.VOL @ f_0

        self._forces = np.zeros([self.mesh.nodes_count, 2])
        self._add_inner_forces()
        self._add_neumann_forces()

    def _add_inner_forces(self):
        for element_id, element in enumerate(self.mesh.elements):
            p_0 = self.mesh.initial_nodes[element[0]]
            p_1 = self.mesh.initial_nodes[element[1]]
            p_2 = self.mesh.initial_nodes[element[2]]

            f_0 = self.inter_forces(*p_0)
            f_1 = self.inter_forces(*p_1)
            f_2 = self.inter_forces(*p_2)

            f_mean = (f_0 + f_1 + f_2) / 3

            self._forces[element[0]] += f_mean / 3 * self.mesh.element_initial_volume[element_id]
            self._forces[element[1]] += f_mean / 3 * self.mesh.element_initial_volume[element_id]
            self._forces[element[2]] += f_mean / 3 * self.mesh.element_initial_volume[element_id]

    def _add_neumann_forces(self):
        for edge in self.mesh.neumann_boundary:
            v_0 = edge[0]
            v_1 = edge[1]

            edge_length = nph.length(
                self.mesh.initial_nodes[v_0], self.mesh.initial_nodes[v_1]
            )
            v_mid = (self.mesh.initial_nodes[v_0] + self.mesh.initial_nodes[v_1]) / 2

            f_neumann = self.outer_forces(*v_mid) * edge_length / 2

            self._forces[v_0] += f_neumann
            self._forces[v_1] += f_neumann
