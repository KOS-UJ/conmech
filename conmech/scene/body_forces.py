from typing import Callable, Optional

import jax
import jax.numpy as jnp
import numpy as np

from conmech.dynamics.dynamics import Dynamics, DynamicsConfiguration
from conmech.helpers import nph
from conmech.helpers.config import SimulationConfig
from conmech.mesh.boundaries_description import BoundariesDescription
from conmech.properties.body_properties import TimeDependentBodyProperties
from conmech.properties.mesh_properties import MeshProperties
from conmech.properties.schedule import Schedule
from conmech.scene.energy_functions import _get_constant_boundary_integral
from conmech.state.body_position import get_surface_per_boundary_node_jax


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


def get_integrated_forces_jax(
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
        simulation_config: SimulationConfig,
        dynamics_config: DynamicsConfiguration,
        boundaries_description: Optional[BoundariesDescription] = None,
    ):
        super().__init__(
            mesh_prop=mesh_prop,
            body_prop=body_prop,
            schedule=schedule,
            simulation_config=simulation_config,
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
        super().prepare()
        self.inner_forces = inner_forces
        self.outer_forces = np.zeros_like(self.initial_nodes)

    def clear_external_factors(self):
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
        neumann_surfaces = jax.jit(
            get_surface_per_boundary_node_jax, static_argnames=["considered_nodes_count"]
        )(
            moved_nodes=self.moved_nodes,  # normalized
            boundary_surfaces=self.neumann_boundary,
            considered_nodes_count=self.nodes_count,
        )
        return neumann_surfaces * self.normalized_outer_forces

    def get_integrated_forces_column_np(self):
        integrated_inner_forces = self.matrices.volume_at_nodes @ self.normalized_inner_forces
        integrated_outer_forces = self.get_normalized_integrated_outer_forces()
        integrated_forces = integrated_inner_forces + integrated_outer_forces
        return nph.stack_column(integrated_forces)  # [self.independent_indices, :])

    def get_integrated_forces_vector_np(self):
        return np.array(self.get_integrated_forces_column_np().reshape(-1), dtype=np.float64)

    def get_normalized_integrated_forces_column_for_jax(self, args):
        integrated_forces = jax.jit(get_integrated_forces_jax)(
            volume_at_nodes_jax=self.matrices.volume_at_nodes_jax,
            normalized_inner_forces=self.normalized_inner_forces,
            integrated_outer_forces=self.get_normalized_integrated_outer_forces(),
        )

        if self.simulation_config.use_constant_contact_integral:
            rhs_contact = jax.jit(
                _get_constant_boundary_integral, static_argnames="use_nonconvex_friction_law"
            )(
                args=args,
                use_nonconvex_friction_law=self.simulation_config.use_nonconvex_friction_law,
            )
            integrated_forces = integrated_forces - rhs_contact

        return nph.stack_column(
            integrated_forces[self.independent_indices, :]
        )  # Skipping Dirichlet nodes
