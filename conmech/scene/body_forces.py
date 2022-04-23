from typing import Callable

import numpy as np

from conmech.dynamics.dynamics import Dynamics
from conmech.helpers import nph
from conmech.properties.body_properties import DynamicBodyProperties
from conmech.properties.mesh_properties import MeshProperties
from conmech.properties.schedule import Schedule
from conmech.solvers.optimization.schur_complement import SchurComplement


def energy(value, lhs, rhs):
    value_vector = nph.stack_column(value)
    first = 0.5 * (lhs @ value_vector) - rhs
    value = first.reshape(-1) @ value_vector
    return value


class BodyForces(Dynamics):
    def __init__(
        self,
        mesh_prop: MeshProperties,
        body_prop: DynamicBodyProperties,
        schedule: Schedule,
        normalize_by_rotation: bool,
        inner_forces: Callable = (lambda _: 0),
        outer_forces: Callable = (lambda _: 0),
        is_dirichlet: Callable = (lambda _: False),
        is_contact: Callable = (lambda _: True),
        create_in_subprocess: bool = False,
        with_lhs: bool = True,
        with_schur: bool = True,
        with_forces: bool = False,
    ):
        super().__init__(
            mesh_prop=mesh_prop,
            body_prop=body_prop,
            schedule=schedule,
            normalize_by_rotation=normalize_by_rotation,
            is_dirichlet=is_dirichlet,
            is_contact=is_contact,
            create_in_subprocess=create_in_subprocess,
            with_lhs=with_lhs,
            with_schur=with_schur,
        )

        self.with_forces = with_forces

        self.forces = None
        if self.with_forces:
            # RHS
            self.inner_forces = inner_forces
            self.outer_forces = outer_forces
            self._forces = np.zeros([self.nodes_count, 2])
            self._add_inner_forces()
            self._add_neumann_forces()

    @property
    def forces_vector(self):
        return nph.stack_column(self._forces[: self.independent_nodes_count, :]).reshape(-1)

    def _add_inner_forces(self):
        for element_id, element in enumerate(self.elements):
            p_0 = self.initial_nodes[element[0]]
            p_1 = self.initial_nodes[element[1]]
            p_2 = self.initial_nodes[element[2]]

            f_0 = self.inner_forces(p_0)
            f_1 = self.inner_forces(p_1)
            f_2 = self.inner_forces(p_2)

            f_mean = (f_0 + f_1 + f_2) / 3

            self._forces[element[0]] += f_mean / 3 * self.element_initial_volume[element_id]
            self._forces[element[1]] += f_mean / 3 * self.element_initial_volume[element_id]
            self._forces[element[2]] += f_mean / 3 * self.element_initial_volume[element_id]

    def _add_neumann_forces(self):
        for edge in self.neumann_boundary:
            v_0 = edge[0]
            v_1 = edge[1]

            edge_length = nph.length(self.initial_nodes[v_0], self.initial_nodes[v_1])
            v_mid = (self.initial_nodes[v_0] + self.initial_nodes[v_1]) / 2

            f_neumann = self.outer_forces(v_mid) * edge_length / 2

            self._forces[v_0] += f_neumann
            self._forces[v_1] += f_neumann

    @property
    def normalized_forces(self):
        return self.normalize_rotate(self.forces)

    def prepare(self, forces: np.ndarray):
        self.forces = forces

    def clear(self):
        self.forces = None

    # def integrate_forces(self):
    #    f_vector = nph.stack_column(self.volume @ forces)

    def get_all_normalized_rhs_np(self, temperature=None):
        normalized_rhs = self.get_normalized_rhs_np(temperature)
        (
            normalized_rhs_boundary,
            normalized_rhs_free,
        ) = SchurComplement.calculate_schur_complement_vector(
            vector=normalized_rhs,
            dimension=self.dimension,
            contact_indices=self.contact_indices,
            free_indices=self.free_indices,
            free_x_free_inverted=self.solver_cache.free_x_free_inverted,
            contact_x_free=self.solver_cache.contact_x_free,
        )
        return normalized_rhs_boundary, normalized_rhs_free

    def get_normalized_rhs_np(self, temperature=None):
        _ = temperature

        displacement_old = self.normalized_displacement_old
        velocity_old = self.normalized_velocity_old

        displacement_old_vector = nph.stack_column(displacement_old)
        velocity_old_vector = nph.stack_column(velocity_old)
        f_vector = nph.stack_column(self.volume @ self.normalized_forces)
        rhs = (
            f_vector
            - (self.viscosity + self.elasticity * self.time_step) @ velocity_old_vector
            - self.elasticity @ displacement_old_vector
        )
        return rhs

    @property
    def input_forces(self):
        return self.normalized_forces
