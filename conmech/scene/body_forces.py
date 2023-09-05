import dataclasses
import enum
from typing import Callable, Optional

import numpy as np

from conmech.helpers import nph
from conmech.helpers.schur_complement_functions import calculate_schur_complement_vector
from conmech.state.body_position import get_surface_per_boundary_node_numba


def energy(value, lhs, rhs):
    value_vector = nph.stack_column(value)
    first = 0.5 * (lhs @ value_vector) - rhs
    value = first.reshape(-1) @ value_vector
    return value


class Field(enum.Enum):
    FORCE = "FORCE"
    TEMPERATURE = "TEMPERATURE"
    ELECTRIC = "ELECTRIC"


@dataclasses.dataclass()
class FieldSource:
    source: Optional[Callable[[np.ndarray, float], np.ndarray]] = None
    cache: Optional[np.ndarray] = None
    timestamp: Optional[float] = None

    def node_source(self, nodes, time: float):
        # pylint: disable=not-callable
        if time != self.timestamp:
            self.cache = np.array(
                [self.source(nodes[i], time) for i in range(len(nodes))]
            )
            self.timestamp = time
        return self.cache


class BodyForces:
    def __init__(
        self,
        body: "Body"
    ):
        self.body = body
        self.inner = FieldSource()
        self.outer = FieldSource()

    def prepare(self, inner_forces: np.ndarray):
        self.inner.cache = inner_forces
        self.outer.cache = np.zeros_like(self.body.mesh.initial_nodes)
        self.inner.timestamp = 0
        self.outer.timestamp = 0

    def clear(self):
        self.inner.cache = None
        self.outer.cache = None

    def get_integrated_inner_forces(self, time: float):
        inner_forces = self.body.state.position.normalize_rotate(
            self.inner.node_source(self.body.mesh.initial_nodes, time))
        # TODO: should be only on boundary!
        return self.body.dynamics.volume_at_nodes @ inner_forces

    def get_integrated_outer_forces(self, time: float):
        neumann_surfaces = get_surface_per_boundary_node_numba(
            boundary_surfaces=self.body.mesh.neumann_boundary,
            considered_nodes_count=self.body.mesh.nodes_count,
            moved_nodes=self.body.state.position.moved_nodes,
        )
        outer_forces = self.outer.node_source(self.body.mesh.initial_nodes, time)
        return neumann_surfaces * outer_forces

    def get_integrated_field_sources_column(self, time: float):
        integrated_inner_forces = self.get_integrated_inner_forces(time)
        integrated_outer_forces = self.get_integrated_outer_forces(time)
        integrated_forces = integrated_inner_forces + integrated_outer_forces
        return nph.stack_column(integrated_forces[:, :])

    def integrate(self, time: float):
        return self.get_integrated_field_sources_column(time).reshape(-1)

    def get_all_normalized_rhs_np(self, temperature=None):
        normalized_rhs = self.get_normalized_rhs_np(temperature)
        (
            normalized_rhs_boundary,
            normalized_rhs_free,
        ) = calculate_schur_complement_vector(
            vector=normalized_rhs,
            dimension=self.body.mesh.dimension,
            contact_indices=self.body.mesh.contact_indices,
            free_indices=self.body.mesh.free_indices,
            free_x_free_inverted=self.body.dynamics.solver_cache.free_x_free_inverted,
            contact_x_free=self.body.dynamics.solver_cache.contact_x_free,
        )
        return normalized_rhs_boundary, normalized_rhs_free

    def get_normalized_rhs_np(self, temperature=None) -> np.ndarray:
        _ = temperature

        displacement_old_vector = nph.stack_column(self.body.state.position.normalized_displacement_old)
        velocity_old_vector = nph.stack_column(self.body.state.position.normalized_velocity_old)
        f_vector = self.get_integrated_field_sources_column(time=0)
        rhs = (
            f_vector
            - (self.body.dynamics.viscosity + self.body.dynamics.elasticity * self.body.dynamics.time_step) @ velocity_old_vector
            - self.body.dynamics.elasticity @ displacement_old_vector
        )
        return rhs
