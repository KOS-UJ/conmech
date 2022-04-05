from typing import Callable
from conmech.boundaries_builder import BoundariesBuilder, BoundariesData

import deep_conmech.simulator.mesh.mesh_builders as mesh_builders
import numba
import numpy as np
from numba import njit

import deep_conmech.simulator.mesh.mesh_builders as mesh_builders
from conmech.properties.mesh_properties import MeshProperties
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
            mesh_data: MeshProperties,
            normalize_by_rotation: bool,
            is_dirichlet: Callable = (lambda _: False),
            is_contact: Callable = (lambda _: True),
            create_in_subprocess: bool = False,
    ):
        self.mesh_data = mesh_data
        self.normalize_by_rotation = normalize_by_rotation

        self.initial_nodes: np.ndarray
        self.elements: np.ndarray
        self.edges: np.ndarray

        self.boundaries_data: BoundariesData

        self.base_seed_indices: np.ndarray
        self.closest_seed_index: int

        self.reinitialize_data(mesh_data, is_dirichlet, is_contact, create_in_subprocess)


    def remesh(self, is_dirichlet, is_contact, create_in_subprocess):
        self.reinitialize_data(self.mesh_data, is_dirichlet, is_contact, create_in_subprocess)


     

    def reinitialize_data(self, mesh_data, is_dirichlet, is_contact, create_in_subprocess):
        input_nodes, input_elements = mesh_builders.build_mesh(
            mesh_data=mesh_data, create_in_subprocess=create_in_subprocess,
        )
        unordered_nodes, unordered_elements = remove_unconnected_nodes_numba(
            input_nodes, input_elements
        )

        self.initial_nodes, self.elements, self.boundaries_data = BoundariesBuilder.identify_boundaries_and_reorder_nodes(unordered_nodes, unordered_elements, is_dirichlet, is_contact)

        self.base_seed_indices, self.closest_seed_index = get_base_seed_indices_numba(
            self.initial_nodes
        )

        edges_matrix = get_edges_matrix(nodes_count=len(self.initial_nodes), elements=self.elements)
        self.edges = get_edges_list_numba(edges_matrix)



    def get_state_dict(self):
        return vars(self)

    def load_state_dict(self, state_dict):
        for key, attr in state_dict.items():
            self.__setattr__(key, attr)



    @property
    def boundary_surfaces(self):
        return self.boundaries_data.boundary_surfaces

    @property
    def contact_surfaces(self):
        return self.boundaries_data.contact_surfaces

    @property
    def neumann_surfaces(self):
        return self.boundaries_data.neumann_surfaces

    @property
    def dirichlet_surfaces(self):
        return self.boundaries_data.dirichlet_surfaces


    @property
    def boundary_internal_indices(self):
        return self.boundaries_data.boundary_internal_indices

        
    @property
    def boundary_nodes_count(self):
        return self.boundaries_data.boundary_nodes_count

    @property
    def contact_nodes_count(self):
        return self.boundaries_data.contact_nodes_count
        
    @property
    def dirichlet_nodes_count(self):
        return self.boundaries_data.dirichlet_nodes_count
        

    @property
    def neumann_nodes_count(self):
        return self.boundaries_data.neumann_nodes_count

    @property
    def independent_nodes_count(self):
        return self.nodes_count - self.dirichlet_nodes_count

    @property
    def free_nodes_count(self):
        return self.independent_nodes_count - self.contact_nodes_count - self.dirichlet_nodes_count # TODO: CHECK




    @property
    def boundary_indices(self):
        return slice(self.boundary_nodes_count)

    @property
    def contact_indices(self):
        return slice(self.contact_nodes_count)

    @property
    def neumann_indices(self):
        return slice(self.contact_nodes_count, self.contact_nodes_count + self.neumann_nodes_count)

    @property
    def dirichlet_indices(self):
        return slice(self.nodes_count - self.dirichlet_nodes_count, self.nodes_count)


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
    def boundary_surfaces_count(self):
        return len(self.boundary_surfaces)

    @property
    def inner_nodes_count(self):
        return self.nodes_count - self.boundary_nodes_count

    @property
    def edges_number(self):
        return len(self.edges)
