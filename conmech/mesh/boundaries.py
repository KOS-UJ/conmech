# CONMECH @ Jagiellonian University in Kraków
#
# Copyright (C) 2023  Piotr Bartman-Szwarc <piotr.bartman@uj.edu.pl>
#
# This program is free software; you can redistribute it and/or
# modify it under the terms of the GNU General Public License
# as published by the Free Software Foundation; either version 3
# of the License, or (at your option) any later version.
#
# This program is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
# GNU General Public License for more details.
#
# You should have received a copy of the GNU General Public License
# along with this program; if not, write to the Free Software
# Foundation, Inc., 51 Franklin Street, Fifth Floor, Boston, MA  02110-1301,
# USA.

from typing import Dict

import numpy as np

from conmech.mesh.boundary import Boundary
from conmech.helpers import nph


def normalize_euclidean_numba(data: np.ndarray) -> np.ndarray:
    norm = np.linalg.norm(data, axis=-1)
    reshaped_norm = norm if data.ndim == 1 else norm.reshape(-1, 1)
    return data / reshaped_norm


def get_unoriented_normals_2d(faces_nodes):
    tail_nodes, head_nodes = faces_nodes[:, 0], faces_nodes[:, 1]

    normal = normalize_euclidean_numba(head_nodes - tail_nodes)
    unoriented_normals = np.array((normal[..., 1], -normal[..., 0])).T
    return tail_nodes, unoriented_normals


def get_unoriented_normals_3d(faces_nodes):
    tail_nodes, head_nodes1, head_nodes2 = [faces_nodes[:, i, :] for i in range(3)]

    unoriented_normals = normalize_euclidean_numba(
        np.cross(head_nodes1 - tail_nodes, head_nodes2 - tail_nodes)
    )
    return tail_nodes, unoriented_normals


def get_boundary_surfaces_normals(nodes, boundary_surfaces, boundary_internal_indices):
    dim = nodes.shape[1]
    faces_nodes = nodes[boundary_surfaces]

    if dim == 2:
        tail_nodes, unoriented_normals = get_unoriented_normals_2d(faces_nodes)
    elif dim == 3:
        tail_nodes, unoriented_normals = get_unoriented_normals_3d(faces_nodes)
    else:
        raise ValueError()

    internal_nodes = nodes[boundary_internal_indices]
    external_orientation = (-1) * np.sign(
        nph.elementwise_dot(internal_nodes - tail_nodes, unoriented_normals, keepdims=True)
    )
    return unoriented_normals * external_orientation


class Boundaries:
    def __init__(self, nodes, boundary_internal_indices: np.ndarray, **kwargs):
        self.boundaries: Dict[str, Boundary] = {}
        for key, value in kwargs.items():
            assert isinstance(key, str)
            assert isinstance(value, Boundary)
            self.boundaries[key] = value

        self.boundary_internal_indices = boundary_internal_indices
        self.surface_normals = get_boundary_surfaces_normals(
            nodes,
            self.boundary_surfaces,
            self.boundary_internal_indices,
        )

    @property
    def boundary_surfaces(self):
        return np.unique(
            np.vstack((self.contact_boundary, self.neumann_boundary, self.dirichlet_boundary)),
            axis=1,
        )

    @property
    def boundary_nodes_count(self):
        return self.contact_nodes_count + self.neumann_nodes_count + self.dirichlet_nodes_count

    @property
    def boundary_indices(self):
        return slice(self.boundary_nodes_count)

    @property
    def contact_boundary(self):
        return self.boundaries["contact"].surfaces

    @property
    def contact_normals(self):
        return self.surface_normals[: self.contact_nodes_count]

    @property
    def neumann_boundary(self):
        return self.boundaries["neumann"].surfaces

    @property
    def dirichlet_boundary(self):
        return self.boundaries["dirichlet"].surfaces

    @property
    def contact_nodes_count(self) -> int:
        return self.boundaries["contact"].node_count

    @property
    def neumann_nodes_count(self) -> int:
        return self.boundaries["neumann"].node_count

    @property
    def dirichlet_nodes_count(self) -> int:
        return self.boundaries["dirichlet"].node_count

    def get_all_boundary_indices(self, boundary, total_node_count, dimension):
        i = self.boundaries[boundary].node_indices
        if isinstance(i, slice):
            for d in range(dimension):
                condition_start = 0
                condition_stop = condition_start + i.stop - i.start
                yield (
                    slice(i.start + d * total_node_count, i.stop + d * total_node_count),
                    slice(condition_start, condition_stop),
                )
        else:
            # assuming node_indices are sorted
            discontinuities = np.concatenate(([0], np.nonzero(np.diff(i) - 1)[0] + 1, [len(i)]))
            starts = i[discontinuities[:-1]]
            stops = i[(discontinuities - 1)[1:]] + 1
            for d in range(dimension):
                condition_start = 0
                for start, stop in zip(starts, stops):
                    condition_stop = condition_start + stop - start
                    yield (
                        slice(start + d * total_node_count, stop + d * total_node_count),
                        slice(condition_start, condition_stop),
                    )
                    condition_start = condition_stop
