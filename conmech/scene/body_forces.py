from typing import Callable, Optional

import jax
import jax.numpy as jnp
import numpy as np

from conmech.dynamics.dynamics import Dynamics, DynamicsConfiguration
from conmech.helpers import nph
from conmech.mesh.boundaries_description import BoundariesDescription
from conmech.properties.body_properties import TimeDependentBodyProperties
from conmech.properties.mesh_properties import MeshProperties
from conmech.properties.schedule import Schedule
from conmech.solvers.optimization.schur_complement import SchurComplement
from conmech.state.body_position import get_surface_per_boundary_node_numba


def energy(value, solver_cache, rhs):
    return energy_vector(nph.stack_column(value), solver_cache, rhs)


def energy_vector(value_vector, lhs, rhs):
    lhs_times_value = lhs @ value_vector
    first = 0.5 * lhs_times_value - rhs
    value = first.reshape(-1) @ value_vector
    return value[0]


def energy_lhs(value, lhs, rhs):
    return energy_vector_lhs(nph.stack_column(value), lhs, rhs)


def energy_vector_lhs(value_vector, lhs, rhs):
    first = 0.5 * (lhs @ value_vector) - rhs
    value = first.reshape(-1) @ value_vector
    return value[0]


@jax.jit
def get_integrated_forces_jax_int(
    volume_at_nodes_jax, normalized_inner_forces, integrated_outer_forces
):
    integrated_inner_forces = volume_at_nodes_jax @ jnp.array(normalized_inner_forces)
    integrated_outer_forces = jnp.array(integrated_outer_forces)
    integrated_forces = integrated_inner_forces + integrated_outer_forces
    return integrated_forces


class BodyForces(Dynamics):
    def __init__(
        self,
        mesh_prop: MeshProperties,
        body_prop: TimeDependentBodyProperties,
        schedule: Schedule,
        dynamics_config: DynamicsConfiguration,
        boundaries_description: Optional[BoundariesDescription] = None,
    ):
        super().__init__(
            mesh_prop=mesh_prop,
            body_prop=body_prop,
            schedule=schedule,
            dynamics_config=dynamics_config,
            boundaries_description=boundaries_description,
        )

        self.inner_forces = None
        self.outer_forces = None

    def set_permanent_forces_by_functions(
        self, inner_forces_function: Callable, outer_forces_function: Callable
    ):
        self.inner_forces = np.array([inner_forces_function(p) for p in self.moved_nodes])
        self.outer_forces = np.array([outer_forces_function(p) for p in self.moved_nodes])

    def prepare(self, inner_forces: np.ndarray):
        self.inner_forces = inner_forces
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

    @property
    def boundary_forces(self):
        return self.normalized_inner_forces[self.boundary_indices]

    @property
    def input_forces(self):
        return self.normalized_inner_forces

    def get_normalized_integrated_outer_forces(self):
        neumann_surfaces = get_surface_per_boundary_node_numba(
            boundary_surfaces=self.neumann_boundary,
            considered_nodes_count=self.nodes_count,
            moved_nodes=self.moved_nodes,  # normalized
        )
        return neumann_surfaces * self.normalized_outer_forces

    def get_integrated_forces_column_np(self):
        integrated_inner_forces = self.matrices.volume_at_nodes @ self.normalized_inner_forces
        integrated_outer_forces = self.get_normalized_integrated_outer_forces()
        integrated_forces = integrated_inner_forces + integrated_outer_forces
        return nph.stack_column(integrated_forces)  # [self.independent_indices, :])

    # def get_integrated_forces_column(self):
    #     integrated_forces = self.get_integrated_inner_forces()
    #     + self.get_integrated_outer_forces()
    #     return nph.stack_column(integrated_forces[:, :])

    def get_integrated_forces_vector_np(self):
        return np.array(self.get_integrated_forces_column_np().reshape(-1), dtype=np.float64)

    def get_normalized_integrated_forces_column_for_jax(self):
        integrated_forces = get_integrated_forces_jax_int(
            volume_at_nodes_jax=self.matrices.volume_at_nodes_jax,
            normalized_inner_forces=self.normalized_inner_forces,
            integrated_outer_forces=self.get_normalized_integrated_outer_forces(),
        )
        return nph.stack_column(integrated_forces)  # [self.independent_indices, :])
        # return neumann_surfaces * self.outer_forces

    def get_all_normalized_rhs_jax(self, temperature=None):
        normalized_rhs = self.get_normalized_rhs_jax(temperature)
        (
            normalized_rhs_boundary,
            normalized_rhs_free,
        ) = SchurComplement.calculate_schur_complement_vector(
            vector=normalized_rhs,
            dimension=self.dimension,
            contact_indices=self.contact_indices,
            free_indices=self.free_indices,
            free_x_free=self.solver_cache.free_x_free,
            contact_x_free=self.solver_cache.contact_x_free,
        )
        return normalized_rhs_boundary, normalized_rhs_free

    def get_normalized_rhs_np(self, temperature=None):
        _ = temperature
        displacement_old_vector = nph.stack_column(self.normalized_displacement_old)
        velocity_old_vector = nph.stack_column(self.normalized_velocity_old)
        f_vector = self.get_integrated_forces_column_np()
        rhs = (
            f_vector
            - (self.matrices.viscosity + self.matrices.elasticity * self.time_step)
            @ velocity_old_vector
            - self.matrices.elasticity @ displacement_old_vector
        )
        return rhs

    def get_normalized_rhs_jax(self, temperature=None):
        _ = temperature
        displacement_old_vector = nph.stack_column(self.normalized_displacement_old)
        velocity_old_vector = nph.stack_column(self.normalized_velocity_old)
        f_vector = self.get_normalized_integrated_forces_column_for_jax()
        rhs = (
            f_vector
            - (self.matrices.viscosity + self.matrices.elasticity * self.time_step)
            @ jnp.array(velocity_old_vector)
            - self.matrices.elasticity @ jnp.array(displacement_old_vector)
        )
        return rhs
