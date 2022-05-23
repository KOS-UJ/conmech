import copy
from ctypes import ArgumentError
from typing import Callable

import numba
import numpy as np

from conmech.helpers import nph
from conmech.mesh.mesh import Mesh
from conmech.properties.mesh_properties import MeshProperties
from conmech.properties.schedule import Schedule


def get_base(nodes, base_seed_indices, closest_seed_index):
    base_seed_initial_nodes = nodes[base_seed_indices]
    base_seed = base_seed_initial_nodes[..., 1, :] - base_seed_initial_nodes[..., 0, :]
    return nph.complete_base(base_seed, closest_seed_index)


def get_unoriented_normals_2d(faces_nodes):
    tail_nodes, head_nodes = faces_nodes[:, 0], faces_nodes[:, 1]

    unoriented_normals = nph.get_tangential_2d(
        nph.normalize_euclidean_numba(head_nodes - tail_nodes)
    )
    return tail_nodes, unoriented_normals


def get_unoriented_normals_3d(faces_nodes):
    tail_nodes, head_nodes1, head_nodes2 = [faces_nodes[:, i, :] for i in range(3)]

    unoriented_normals = nph.normalize_euclidean_numba(
        np.cross(head_nodes1 - tail_nodes, head_nodes2 - tail_nodes)
    )
    return tail_nodes, unoriented_normals


def get_boundary_surfaces_normals(moved_nodes, boundary_surfaces, boundary_internal_indices):
    dim = moved_nodes.shape[1]
    faces_nodes = moved_nodes[boundary_surfaces]

    if dim == 2:
        tail_nodes, unoriented_normals = get_unoriented_normals_2d(faces_nodes)
    elif dim == 3:
        tail_nodes, unoriented_normals = get_unoriented_normals_3d(faces_nodes)
    else:
        raise ArgumentError

    internal_nodes = moved_nodes[boundary_internal_indices]
    external_orientation = (-1) * np.sign(
        nph.elementwise_dot(internal_nodes - tail_nodes, unoriented_normals, keepdims=True)
    )
    return unoriented_normals * external_orientation


@numba.njit
def get_boundary_nodes_normals_numba(
    boundary_surfaces, boundary_nodes_count, boundary_surfaces_normals
):
    dim = boundary_surfaces_normals.shape[1]
    boundary_normals = np.zeros((boundary_nodes_count, dim), dtype=np.float64)
    node_faces_count = np.zeros((boundary_nodes_count, 1), dtype=np.int32)

    for i, boundary_surface in enumerate(boundary_surfaces):
        boundary_normals[boundary_surface] += boundary_surfaces_normals[i]
        node_faces_count[boundary_surface] += 1

    boundary_normals /= node_faces_count

    boundary_normals = nph.normalize_euclidean_numba(boundary_normals)
    return boundary_normals


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


class BodyPosition(Mesh):
    def __init__(
        self,
        mesh_prop: MeshProperties,
        schedule: Schedule,
        normalize_by_rotation: bool,
        is_dirichlet: Callable = (lambda _: False),
        is_contact: Callable = (lambda _: True),
        create_in_subprocess: bool = False,
    ):
        super().__init__(
            mesh_prop=mesh_prop,
            is_dirichlet=is_dirichlet,
            is_contact=is_contact,
            create_in_subprocess=create_in_subprocess,
        )

        self.schedule = schedule
        self.normalize_by_rotation = normalize_by_rotation
        self.displacement_old = np.zeros_like(self.initial_nodes)
        self.velocity_old = np.zeros_like(self.initial_nodes)
        self.acceleration_old = np.zeros_like(self.initial_nodes)

    def set_acceleration_old(self, acceleration):
        self.acceleration_old = acceleration

    def set_velocity_old(self, velocity):
        self.velocity_old = velocity

    def set_displacement_old(self, displacement):
        self.displacement_old = displacement

    @property
    def time_step(self):
        return self.schedule.time_step

    def get_copy(self):
        return copy.deepcopy(self)

    def iterate_self(self, acceleration, temperature=None):
        _ = temperature
        velocity = self.velocity_old + self.time_step * acceleration
        displacement = self.displacement_old + self.time_step * velocity

        self.set_displacement_old(displacement)
        self.set_velocity_old(velocity)
        self.set_acceleration_old(acceleration)

        return self

    @property
    def moved_base(self):
        return get_base(self.moved_nodes, self.base_seed_indices, self.closest_seed_index)

    def normalize_rotate(self, vectors):
        if not self.normalize_by_rotation:
            return vectors
        return nph.get_in_base(vectors, self.moved_base)

    def denormalize_rotate(self, vectors):
        if not self.normalize_by_rotation:
            return vectors
        return nph.get_in_base(vectors, np.linalg.inv(self.moved_base))

    def normalize_shift_and_rotate(self, vectors):
        return self.normalize_rotate(self.normalize_shift(vectors))

    @property
    def moved_nodes(self):
        return self.initial_nodes + self.displacement_old

    @property
    def normalized_nodes(self):
        return self.normalize_shift_and_rotate(self.moved_nodes)

    @property
    def boundary_nodes(self):
        return self.moved_nodes[self.boundary_indices]

    @property
    def normalized_boundary_nodes(self):
        return self.normalized_nodes[self.boundary_indices]

    def get_normalized_boundary_normals(self):
        return self.normalize_rotate(self.get_boundary_normals())

    @property
    def normalized_a_old(self):
        return self.normalize_rotate(self.acceleration_old)

    @property
    def mean_moved_nodes(self):
        return np.mean(self.moved_nodes, axis=0)

    @property
    def edges_moved_nodes(self):
        return self.moved_nodes[self.edges]

    @property
    def edges_normalized_nodes(self):
        return self.normalized_nodes[self.edges]

    @property
    def elements_normalized_nodes(self):
        return self.normalized_nodes[self.elements]

    @property
    def boundary_centers(self):
        return np.mean(self.moved_nodes[self.boundary_surfaces], axis=1)

    @property
    def rotated_velocity_old(self):
        return self.normalize_rotate(self.velocity_old)

    @property
    def normalized_velocity_old(self):
        return self.normalize_shift_and_rotate(self.velocity_old)

    @property
    def normalized_displacement_old(self):
        return self.normalized_nodes - self.normalized_initial_nodes

    @property
    def origin_displacement_old(self):
        return self.denormalize_rotate(self.normalized_displacement_old)

    def get_boundary_normals(self):
        boundary_surfaces_normals = get_boundary_surfaces_normals(
            self.moved_nodes, self.boundary_surfaces, self.boundary_internal_indices
        )
        return get_boundary_nodes_normals_numba(
            self.boundary_surfaces, self.boundary_nodes_count, boundary_surfaces_normals
        )

    def get_surface_per_boundary_node(self):
        return get_surface_per_boundary_node_numba(
            boundary_surfaces=self.boundary_surfaces,
            considered_nodes_count=self.boundary_nodes_count,
            moved_nodes=self.moved_nodes,
        )

    @property
    def input_velocity_old(self):
        return self.normalized_velocity_old

    @property
    def input_displacement_old(self):
        return self.normalized_displacement_old
