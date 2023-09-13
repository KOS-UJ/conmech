import dataclasses
import enum
from typing import Callable, Optional
from ctypes import ArgumentError

import numpy as np
import numba

from conmech.helpers import nph


@numba.njit
def get_surface_per_boundary_node_numba(boundary_surfaces, considered_nodes_count, moved_nodes):
    surface_per_boundary_node = np.zeros((considered_nodes_count, 1), dtype=np.float64)

    for boundary_surface in boundary_surfaces:
        face_nodes = moved_nodes[boundary_surface]
        surface_per_boundary_node[boundary_surface] += element_volume_part_numba(face_nodes)

    return surface_per_boundary_node


@numba.njit
def element_volume_part_numba(face_nodes):
    dim = face_nodes.shape[1]
    nodes_count = face_nodes.shape[0]
    if dim == 2:
        volume = nph.euclidean_norm_numba(face_nodes[0] - face_nodes[1])
    elif dim == 3:
        volume = 0.5 * nph.euclidean_norm_numba(
            np.cross(face_nodes[1] - face_nodes[0], face_nodes[2] - face_nodes[0])
        )
    else:
        raise ArgumentError
    return volume / nodes_count


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
            self.cache = np.array([self.source(nodes[i], time) for i in range(len(nodes))])
            self.timestamp = time
        return self.cache


class BodyForces:
    def __init__(self, body: "Body"):
        self.body = body
        self.inner = FieldSource()
        self.outer = FieldSource()

    def clear(self):
        self.inner.cache = None
        self.outer.cache = None

    def get_integrated_inner_forces(self, time: float):
        inner_forces = self.inner.node_source(self.body.mesh.nodes, time)
        # TODO: should be only on boundary!
        return self.body.dynamics.volume_at_nodes @ inner_forces

    def get_integrated_outer_forces(self, time: float):
        neumann_surfaces = get_surface_per_boundary_node_numba(
            boundary_surfaces=self.body.mesh.neumann_boundary,
            considered_nodes_count=self.body.mesh.nodes_count,
            moved_nodes=self.body.mesh.nodes,
        )
        outer_forces = self.outer.node_source(self.body.mesh.nodes, time)
        return neumann_surfaces * outer_forces

    def get_integrated_field_sources_column(self, time: float):
        integrated_inner_forces = self.get_integrated_inner_forces(time)
        integrated_outer_forces = self.get_integrated_outer_forces(time)
        integrated_forces = integrated_inner_forces + integrated_outer_forces
        return nph.stack_column(integrated_forces[:, :])

    def integrate(self, time: float):
        return self.get_integrated_field_sources_column(time).reshape(-1)
