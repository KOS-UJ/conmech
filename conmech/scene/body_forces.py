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

        self.inner_forces: Optional[Callable[[np.ndarray, float], np.ndarray]] = None
        self._node_inner_forces = None
        self._inner_forces_time = None
        self.outer_forces: Optional[Callable[[np.ndarray, float], np.ndarray]] = None
        self._node_outer_forces = None
        self._outer_forces_time = None

    def prepare(self, inner_forces: np.ndarray):
        self._node_inner_forces = inner_forces
        self._node_outer_forces = np.zeros_like(self.mesh.initial_nodes)
        self._inner_forces_time = 0
        self._outer_forces_time = 0

    def clear(self):
        self._node_inner_forces = None
        self._node_outer_forces = None

    def node_inner_forces(self, time: float):
        # TODO handle set self.inner_forces
        # pylint: disable=not-callable
        if time != self._inner_forces_time:
            self._node_inner_forces = np.array(
                [self.inner_forces(p, time) for p in self.mesh.initial_nodes]
            )
            self._inner_forces_time = time
        return self._node_inner_forces

    def node_outer_forces(self, time: float):
        # TODO handle set self.inner_forces
        # pylint: disable=not-callable
        if time != self._outer_forces_time:
            self._node_outer_forces = np.array(
                [self.outer_forces(p, time) for p in self.mesh.initial_nodes]
            )  # TODO: should be only on boundary!
            self._outer_forces_time = time
        return self._node_outer_forces

    def normalized_inner_forces(self, time: float = 0):
        return self.normalize_rotate(self.node_inner_forces(time))

    def get_integrated_inner_forces(self, time):
        return self.volume_at_nodes @ self.normalized_inner_forces(time)

    def get_integrated_outer_forces(self, time):
        neumann_surfaces = get_surface_per_boundary_node_numba(
            boundary_surfaces=self.mesh.neumann_boundary,
            considered_nodes_count=self.mesh.nodes_count,
            moved_nodes=self.moved_nodes,
        )
        return neumann_surfaces * self.node_outer_forces(time)

    def get_integrated_forces_column(self, time: float):
        integrated_forces = self.get_integrated_inner_forces(
            time
        ) + self.get_integrated_outer_forces(time)
        return nph.stack_column(integrated_forces[:, :])

    def get_integrated_forces_vector(self, time: float):
        return self.get_integrated_forces_column(time).reshape(-1)

    def get_all_normalized_rhs_np(self, temperature=None):
        normalized_rhs = self.get_normalized_rhs_np(temperature)
        (normalized_rhs_boundary, normalized_rhs_free,) = calculate_schur_complement_vector(
            vector=normalized_rhs,
            dimension=self.mesh.dimension,
            contact_indices=self.mesh.contact_indices,
            free_indices=self.mesh.free_indices,
            free_x_free_inverted=self.solver_cache.free_x_free_inverted,
            contact_x_free=self.solver_cache.contact_x_free,
        )
        return normalized_rhs_boundary, normalized_rhs_free

    def get_normalized_rhs_np(self, temperature=None) -> np.ndarray:
        _ = temperature

        displacement_old_vector = nph.stack_column(self.normalized_displacement_old)
        velocity_old_vector = nph.stack_column(self.normalized_velocity_old)
        f_vector = self.get_integrated_forces_column(time=0)
        rhs = (
            f_vector
            - (self.viscosity + self.elasticity * self.time_step) @ velocity_old_vector
            - self.elasticity @ displacement_old_vector
        )
        return rhs
