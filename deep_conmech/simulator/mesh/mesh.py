from argparse import ArgumentError
from typing import Callable

import numba
import numpy as np
from numba import njit

import deep_conmech.simulator.mesh.mesh_builders as mesh_builders
from conmech.dataclass.mesh_data import MeshData
from conmech.helpers import nph


@njit
def get_edges_matrix(nodes_count, elements):
    edges_matrix = np.zeros((nodes_count, nodes_count), dtype=numba.int32)
    element_vertices_number = len(elements[0])
    for element in elements:  # TODO: prange?
        for i in range(element_vertices_number):
            for j in range(element_vertices_number):
                if i != j:
                    edges_matrix[element[i], element[j]] += 1.0
    return edges_matrix


@njit
def get_edges_list_numba(edges_matrix):
    nodes_count = edges_matrix.shape[0]
    edges = np.array(
        [
            (i, j)
            for i, j in np.ndindex((nodes_count, nodes_count))
            if j > i and edges_matrix[i, j] > 0
        ],
        dtype=numba.int64,
    )
    return edges


@njit
def remove_unconnected_nodes_numba(nodes, elements):
    nodes_count = len(nodes)
    index = 0
    while index < nodes_count:
        if index in elements:
            index += 1
        else:
            nodes = np.vstack((nodes[:index], nodes[index + 1:]))
            for i in range(elements.shape[0]):
                for j in range(elements.shape[1]):
                    if elements[i, j] > index:
                        elements[i, j] -= 1
            nodes_count -= 1
    return nodes, elements


@njit
def move_boundary_nodes_to_start_numba(
        unordered_points,
        unordered_elements,
        unordered_boundary_indices,
        unordered_contact_indices,  # TODO: move to the top
):
    nodes_count = len(unordered_points)
    boundary_nodes_count = len(unordered_boundary_indices)

    points = np.zeros((nodes_count, unordered_points.shape[1]))
    elements = -(unordered_elements.copy() + 1)

    boundary_index = 0
    inner_index = nodes_count - 1
    for old_index in range(nodes_count):
        point = unordered_points[old_index]

        if old_index in unordered_boundary_indices:
            new_index = boundary_index
            boundary_index += 1
        else:
            new_index = inner_index
            inner_index -= 1

        points[new_index] = point
        elements = np.where(elements == -(old_index + 1), new_index, elements)

    return points, elements, boundary_nodes_count


@njit
def list_all_faces_numba(sorted_elements):
    elements_count, element_size = sorted_elements.shape
    dim = element_size - 1
    faces = np.zeros((element_size * elements_count, dim), dtype=np.int64)
    opposing_indices = np.zeros((element_size * elements_count), dtype=np.int64)
    i = 0
    for j in range(element_size):
        faces[i: i + elements_count, :j] = sorted_elements[:, :j]
        faces[i: i + elements_count, j:dim] = sorted_elements[:, j + 1: element_size]
        opposing_indices[i: i + elements_count] = sorted_elements[:, j]
        i += elements_count
    return faces, opposing_indices


def extract_unique_elements(elements, opposing_indices):
    _, indices, count = np.unique(
        elements, axis=0, return_index=True, return_counts=True
    )
    unique_indices = indices[count == 1]
    return elements[unique_indices], opposing_indices[unique_indices]


def make_get_contact_mask_numba(is_contact):
    @njit
    def get_contact_mask_numba(nodes, boundary_faces):
        return np.array(
            [
                i
                for i in range(len(boundary_faces))
                if np.all(
                np.array([is_contact(node) for node in nodes[boundary_faces[i]]])
            )
            ]
        )

    return get_contact_mask_numba


def get_boundary_faces(nodes, elements, is_contact):
    elements.sort(axis=1)
    faces, opposing_indices = list_all_faces_numba(sorted_elements=elements)
    boundary_faces, boundary_internal_indices = extract_unique_elements(
        faces, opposing_indices
    )
    boundary_indices = np.unique(boundary_faces.flatten(), axis=0)

    get_contact_mask_numba = make_get_contact_mask_numba(njit(is_contact))
    mask = get_contact_mask_numba(nodes, boundary_faces)
    contact_faces = boundary_faces[mask]
    contact_indices = np.unique(contact_faces.flatten(), axis=0)

    return boundary_faces, boundary_indices, contact_indices, boundary_internal_indices


def reorder_boundary_nodes(nodes, elements, is_contact):
    _, boundary_indices, contact_indices, _ = get_boundary_faces(
        nodes, elements, is_contact
    )
    nodes, elements, _ = move_boundary_nodes_to_start_numba(
        nodes, elements, boundary_indices, contact_indices
    )
    return nodes, elements


@njit
def get_closest_to_axis_numba(nodes, variable):
    min_error = 1.0
    final_i, final_j = 0, 0
    nodes_count = len(nodes)
    for i in range(nodes_count):
        for j in range(i + 1, nodes_count):
            error = nph.euclidean_norm_numba(
                np.delete(nodes[i], variable) - np.delete(nodes[j], variable)
            )
            if error < min_error:
                min_error, final_i, final_j = error, i, j

    correct_order = nodes[final_i, variable] < nodes[final_j, variable]
    indices = (final_i, final_j) if correct_order else (final_j, final_i)
    return np.array([min_error, indices[0], indices[1]])


@njit
def get_base_seed_indices_numba(nodes):
    dim = nodes.shape[1]
    base_seed_indices = np.zeros((dim, 2), dtype=np.int64)
    errors = np.zeros(dim)
    for i in range(dim):
        result = get_closest_to_axis_numba(nodes, i)
        errors[i] = result[0]
        base_seed_indices[i] = result[1:].astype(np.int64)
    return base_seed_indices, np.argmin(errors)


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


def get_boundary_faces_normals(moved_nodes, boundary_faces, boundary_internal_indices):
    dim = moved_nodes.shape[1]
    faces_nodes = moved_nodes[boundary_faces]

    if dim == 2:
        tail_nodes, unoriented_normals = get_unoriented_normals_2d(faces_nodes)
    elif dim == 3:
        tail_nodes, unoriented_normals = get_unoriented_normals_3d(faces_nodes)
    else:
        raise ArgumentError

    internal_nodes = moved_nodes[boundary_internal_indices]
    external_orientation = (-1) * np.sign(
        nph.elementwise_dot(
            internal_nodes - tail_nodes, unoriented_normals, keepdims=True
        )
    )
    return unoriented_normals * external_orientation


@njit
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


@njit
def get_boundary_nodes_data_numba(
        boundary_faces_normals, boundary_faces, boundary_nodes_count, moved_nodes
):
    dim = boundary_faces_normals.shape[1]
    boundary_normals = np.zeros((boundary_nodes_count, dim), dtype=np.float64)
    boundary_nodes_volume = np.zeros((boundary_nodes_count, 1), dtype=np.float64)

    for i in range(boundary_nodes_count):
        node_faces_count = 0
        for j in range(len(boundary_faces)):
            if np.any(boundary_faces[j] == i):
                node_faces_count += 1
                boundary_normals[i] += boundary_faces_normals[j]

                face_nodes = moved_nodes[boundary_faces[j]]
                boundary_nodes_volume[i] += element_volume_part_numba(face_nodes)

        boundary_normals[i] /= node_faces_count

    boundary_normals = nph.normalize_euclidean_numba(boundary_normals)
    return boundary_normals, boundary_nodes_volume


class Mesh:
    def __init__(
            self,
            mesh_data: MeshData,
            normalize_by_rotation: bool,
            is_dirichlet: Callable = (lambda _: False),
            is_contact: Callable = (lambda _: True),
            create_in_subprocess: bool = False,
    ):
        self.mesh_data = mesh_data
        self.normalize_by_rotation = normalize_by_rotation
        self.is_dirichlet = is_dirichlet
        self.is_contact = is_contact
        self.create_in_subprocess = create_in_subprocess

        self.set_mesh()

    def remesh(self):
        self.set_mesh()

    def set_mesh(self):
        input_nodes, input_elements = mesh_builders.build_mesh(
            mesh_data=self.mesh_data, create_in_subprocess=self.create_in_subprocess,
        )
        unordered_nodes, unordered_elements = remove_unconnected_nodes_numba(
            input_nodes, input_elements
        )

        self.reorganize_boundaries(unordered_nodes, unordered_elements)

        self.base_seed_indices, self.closest_seed_index = get_base_seed_indices_numba(
            self.initial_nodes
        )

        self.edges_matrix = get_edges_matrix(self.nodes_count, self.elements)
        self.edges = get_edges_list_numba(self.edges_matrix)

        self.u_old = np.zeros_like(self.initial_nodes)
        self.v_old = np.zeros_like(self.initial_nodes)
        self.a_old = np.zeros_like(self.initial_nodes)

        self.clear()

    def reorganize_boundaries(self, unordered_nodes, unordered_elements):
        (
            self.initial_nodes,
            self.elements,
        ) = reorder_boundary_nodes(unordered_nodes, unordered_elements, self.is_contact)
        (
            self.boundary_faces,
            _boundary_indices,
            _contact_indices,
            self.boundary_internal_indices,
        ) = get_boundary_faces(self.initial_nodes, self.elements, self.is_contact)

        self.independent_nodes_count = self.nodes_count
        self.contact_nodes_count = len(_contact_indices)
        self.boundary_nodes_count = len(_boundary_indices)
        self.free_nodes_count = self.independent_nodes_count - self.contact_nodes_count

        self.boundaries = None

        if not np.array_equal(
                _boundary_indices, range(self.boundary_nodes_count)
        ):
            raise ValueError("Bad boundary ordering")

    @property
    def contact_indices(self):
        return slice(self.contact_nodes_count)

    @property
    def boundary_indices(self):
        return slice(self.boundary_nodes_count)

    @property
    def independent_indices(self):
        return slice(self.independent_nodes_count)

    @property
    def free_indices(self):
        return slice(self.contact_nodes_count, self.independent_nodes_count)

    @property
    def dimension(self):
        return self.mesh_data.dimension

    def set_a_old(self, a):
        self.a_old = a

    def set_v_old(self, v):
        self.v_old = v

    def set_u_old(self, u):
        self.u_old = u

    def prepare(self):
        boundary_faces_normals = get_boundary_faces_normals(
            self.moved_nodes, self.boundary_faces, self.boundary_internal_indices
        )
        (
            self.boundary_normals,
            self.boundary_nodes_volume,
        ) = get_boundary_nodes_data_numba(
            boundary_faces_normals,
            self.boundary_faces,
            self.boundary_nodes_count,
            self.moved_nodes,
        )

    def clear(self):
        self.boundary_normals = None
        self.boundary_nodes_volume = None

    @property
    def boundary_nodes(self):
        return self.moved_nodes[self.boundary_indices]

    @property
    def normalized_boundary_nodes(self):
        return self.normalized_points[self.boundary_indices]

    @property
    def normalized_boundary_normals(self):
        return self.normalize_rotate(self.boundary_normals)

    @property
    def normalized_a_old(self):
        return self.normalize_rotate(self.a_old)

    @property
    def mean_moved_nodes(self):
        return np.mean(self.moved_nodes, axis=0)

    @property
    def mean_initial_nodes(self):
        return np.mean(self.initial_nodes, axis=0)

    @property
    def normalized_points(self):
        return self.normalize_rotate(self.moved_nodes - self.mean_moved_nodes)

    @property
    def normalized_initial_nodes(self):
        return self.initial_nodes - self.mean_initial_nodes

    @property
    def rotated_v_old(self):
        return self.normalize_rotate(self.v_old)

    @property
    def normalized_v_old(self):
        return self.normalize_rotate(self.v_old - np.mean(self.v_old, axis=0))

    @property
    def normalized_u_old(self):
        return self.normalized_points - self.normalized_initial_nodes

    @property
    def origin_u_old(self):
        return self.denormalize_rotate(self.normalized_u_old)

    @property
    def moved_nodes(self):
        return self.initial_nodes + self.u_old

    @property
    def moved_base(self):
        return get_base(
            self.moved_nodes, self.base_seed_indices, self.closest_seed_index
        )

    def normalize_rotate(self, vectors):
        return (
            nph.get_in_base(vectors, self.moved_base)
            if self.normalize_by_rotation
            else vectors
        )

    def denormalize_rotate(self, vectors):
        return nph.get_in_base(vectors, np.linalg.inv(self.moved_base))

    @property
    def edges_moved_nodes(self):
        return self.moved_nodes[self.edges]

    @property
    def edges_normalized_points(self):
        return self.normalized_points[self.edges]

    @property
    def elements_normalized_points(self):
        return self.normalized_points[self.elements]

    @property
    def nodes_count(self):
        return len(self.initial_nodes)

    @property
    def boundary_faces_count(self):
        return len(self.boundary_faces)

    @property
    def inner_nodes_count(self):
        return self.nodes_count - self.boundary_nodes_count

    @property
    def boundary_centers(self):
        return np.mean(self.moved_nodes[self.boundary_faces], axis=1)

    @property
    def edges_number(self):
        return len(self.edges)
