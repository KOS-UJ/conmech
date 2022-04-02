from ctypes import ArgumentError
from typing import Callable

import deep_conmech.simulator.mesh.mesh_builders as mesh_builders
import numba
import numpy as np
from conmech.dataclass.mesh_data import MeshData
from conmech.features.boundaries import Boundaries
from conmech.helpers import nph
from numba import njit


@njit
def get_edges_matrix(nodes_count:int, elements:np.ndarray):
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
    return base_seed_indices, int(np.argmin(errors))




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
        self.create_in_subprocess = create_in_subprocess
        self.is_dirichlet = is_dirichlet
        self.is_contact = is_contact

        self.state_dict = self.get_state_dict(self.is_dirichlet, self.is_contact)

    def remesh(self):
        self.state_dict = self.get_state_dict(self.is_dirichlet, self.is_contact)

    def get_state_dict(self, is_dirichlet, is_contact):
        state_dict = dict() 

        input_nodes, input_elements = mesh_builders.build_mesh(
            mesh_data=self.mesh_data, create_in_subprocess=self.create_in_subprocess,
        )
        unordered_nodes, unordered_elements = remove_unconnected_nodes_numba(
            input_nodes, input_elements
        )

        self.reorganize_boundaries(state_dict, unordered_nodes, unordered_elements, is_dirichlet, is_contact)

        state_dict["base_seed_indices"], state_dict["closest_seed_index"] = get_base_seed_indices_numba(
            state_dict["initial_nodes"]
        )

        edges_matrix = get_edges_matrix(nodes_count=len(state_dict["initial_nodes"]), elements=state_dict["elements"])
        state_dict["edges"] = get_edges_list_numba(edges_matrix)

        return state_dict

    def reorganize_boundaries(self, state_dict, unordered_nodes, unordered_elements, is_dirichlet, is_contact):
        (
            state_dict["initial_nodes"],
            state_dict["elements"],
        ) = reorder_boundary_nodes(unordered_nodes, unordered_elements, is_contact)
        (
            state_dict["boundary_faces"],
            _boundary_indices,
            _contact_indices,
            state_dict["boundary_internal_indices"],
        ) = get_boundary_faces(state_dict["initial_nodes"], state_dict["elements"], is_contact)

        state_dict["contact_nodes_count"] = len(_contact_indices)
        state_dict["dirichlet_nodes_count"] = 0
        state_dict["boundary_nodes_count"] = len(_boundary_indices)

        if not np.array_equal(
            _boundary_indices, range(state_dict["boundary_nodes_count"])
        ):
            raise ValueError("Bad boundary ordering")

    
    

    @property
    def edges(self):
        return self.state_dict["edges"]

    @property
    def initial_nodes(self):
        return self.state_dict["initial_nodes"]

    @property
    def boundary_nodes_count(self):
        return self.state_dict["boundary_nodes_count"]

    @property
    def boundary_faces(self):
        return self.state_dict["boundary_faces"]

    @property
    def boundary_internal_indices(self):
        return self.state_dict["boundary_internal_indices"]

    @property
    def base_seed_indices(self):
        return self.state_dict["base_seed_indices"]

    @property
    def closest_seed_index(self):
        return self.state_dict["closest_seed_index"]

    @property
    def dirichlet_nodes_count(self):
        return self.state_dict["dirichlet_nodes_count"]

    @property
    def contact_nodes_count(self):
        return self.state_dict["contact_nodes_count"]

    @property
    def elements(self):
        return self.state_dict["elements"]

    @property
    def boundaries(self):
        return self.state_dict["boundaries"]
                


    @property
    def independent_nodes_count(self):
        return self.nodes_count - self.dirichlet_nodes_count

    @property
    def free_nodes_count(self):
        return self.independent_nodes_count - self.contact_nodes_count # neumann



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



    @property
    def mean_initial_nodes(self):
        return np.mean(self.initial_nodes, axis=0)


    @property
    def normalized_initial_nodes(self):
        return self.initial_nodes - self.mean_initial_nodes





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
    def edges_number(self):
        return len(self.edges)
