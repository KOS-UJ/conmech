from typing import Callable

import numpy as np

from conmech.dynamics.dynamics import Dynamics
from conmech.helpers import nph
from conmech.properties.body_properties import DynamicBodyProperties
from conmech.properties.mesh_properties import MeshProperties
from conmech.properties.schedule import Schedule
from conmech.solvers.optimization.schur_complement import SchurComplement
from conmech.state.body_position import get_surface_per_boundary_node_numba


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
        is_dirichlet: Callable = (lambda _: False),
        is_contact: Callable = (lambda _: True),
        create_in_subprocess: bool = False,
        with_lhs: bool = True,
        with_schur: bool = True,
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

        self.inner_forces = None
        self.outer_forces = None

    def set_permanent_forces_by_functions(
        self, inner_forces_function: Callable, outer_forces_function: Callable
    ):
        self.inner_forces = np.array([inner_forces_function(p) for p in self.moved_nodes])
        self.outer_forces = np.array([outer_forces_function(p) for p in self.moved_nodes])

    def prepare(self, inner_force_at_node: np.ndarray):
        self.inner_forces = inner_force_at_node
        self.outer_forces = np.zeros_like(self.initial_nodes)

    def clear(self):
        self.inner_forces = None
        self.outer_forces = None

    @property
    def normalized_inner_forces(self):
        return self.normalize_rotate(self.inner_forces)

    @property
    def normalized_outer_forces(self):
        return self.normalize_rotate(self.outer_forces)

    def get_integrated_inner_forces(self):
        return self.volume_at_nodes @ self.normalized_inner_forces

    def get_integrated_outer_forces(self):
        neumann_surfaces = get_surface_per_boundary_node_numba(
            boundary_surfaces=self.neumann_boundary,
            considered_nodes_count=self.nodes_count,
            moved_nodes=self.moved_nodes,
        )
        return neumann_surfaces * self.outer_forces

    def get_integrated_forces_column(self):
        integrated_forces = self.get_integrated_inner_forces() + self.get_integrated_outer_forces()
        return nph.stack_column(integrated_forces[self.independent_indices, :])

    def get_integrated_forces_vector(self):
        return self.get_integrated_forces_column().reshape(-1)

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
        f_vector = self.get_integrated_forces_column()
        rhs = (
            f_vector
            - (self.viscosity + self.elasticity * self.time_step) @ velocity_old_vector
            - self.elasticity @ displacement_old_vector
        )
        return rhs
