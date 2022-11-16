from typing import Callable, Optional

import numpy as np

from conmech.dynamics.dynamics import Dynamics, DynamicsConfiguration
from conmech.helpers import nph
from conmech.mesh.boundaries_description import BoundariesDescription
from conmech.properties.body_properties import BodyProperties
from conmech.properties.mesh_properties import MeshProperties
from conmech.properties.schedule import Schedule
from conmech.helpers.schur_complement_functions import calculate_schur_complement_vector
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
        body_prop: BodyProperties,
        schedule: Schedule,
        dynamics_config: DynamicsConfiguration,
        boundaries_description: BoundariesDescription,
    ):
        super().__init__(
            mesh_prop=mesh_prop,
            body_prop=body_prop,
            schedule=schedule,
            dynamics_config=dynamics_config,
            boundaries_description=boundaries_description,
        )

        self.inner_forces: Optional[np.ndarray] = None
        self.outer_forces: Optional[np.ndarray] = None

    def set_permanent_forces_by_functions(
        self, inner_forces_function: Callable, outer_forces_function: Callable
    ) -> None:
        self.inner_forces = np.array([inner_forces_function(p) for p in self.moved_nodes])
        self.outer_forces = np.array([outer_forces_function(p) for p in self.moved_nodes])

    def prepare(self, inner_forces: np.ndarray) -> None:
        self.inner_forces = inner_forces
        self.outer_forces = np.zeros_like(self.mesh.initial_nodes)

    def clear(self) -> None:
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

    def get_integrated_outer_forces(self) -> np.ndarray:
        neumann_surfaces = get_surface_per_boundary_node_numba(
            boundary_surfaces=self.mesh.neumann_boundary,
            considered_nodes_count=self.mesh.nodes_count,
            moved_nodes=self.moved_nodes,
        )
        return neumann_surfaces * self.outer_forces

    def get_integrated_forces_column(self):
        integrated_forces = self.get_integrated_inner_forces() + self.get_integrated_outer_forces()
        return nph.stack_column(integrated_forces[self.mesh.independent_indices, :])

    def get_integrated_forces_vector(self):
        return self.get_integrated_forces_column().reshape(-1)

    def get_all_normalized_rhs_np(self, temperature=None):
        normalized_rhs = self.get_normalized_rhs_np(temperature)
        (
            normalized_rhs_boundary,
            normalized_rhs_free,
        ) = calculate_schur_complement_vector(
            vector=normalized_rhs,
            dimension=self.mesh.dimension,
            contact_indices=self.mesh.contact_indices,
            free_indices=self.mesh.free_indices,
            free_x_free_inverted=self.solver_cache.free_x_free_inverted,
            contact_x_free=self.solver_cache.contact_x_free,
        )
        return normalized_rhs_boundary, normalized_rhs_free

    def get_normalized_rhs_np(self, temperature=None):
        _ = temperature

        displacement_old_vector = nph.stack_column(self.normalized_displacement_old)
        velocity_old_vector = nph.stack_column(self.normalized_velocity_old)
        f_vector = self.get_integrated_forces_column()
        rhs = (
            f_vector
            - (self.viscosity + self.elasticity * self.time_step) @ velocity_old_vector
            - self.elasticity @ displacement_old_vector
        )
        return rhs
