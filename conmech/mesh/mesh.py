from typing import Callable

import numba
import numpy as np

from conmech.helpers import nph
from conmech.mesh import mesh_builders
from conmech.mesh.boundaries_description import BoundariesDescription
from conmech.mesh.boundaries_factory import BoundariesFactory
from conmech.mesh.boundaries import Boundaries
from conmech.properties.mesh_properties import MeshProperties


@numba.njit
def get_edges_matrix(nodes_count: int, elements: np.ndarray):
    edges_matrix = np.zeros((nodes_count, nodes_count), dtype=numba.int32)
    element_vertices_number = len(elements[0])
    for element in elements:  # TODO: #65 prange?
        for i in range(element_vertices_number):
            for j in range(element_vertices_number):
                if i != j:
                    edges_matrix[element[i], element[j]] += 1.0
    return edges_matrix


@numba.njit
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


@numba.njit
def remove_unconnected_nodes_numba(nodes, elements):
    nodes_count = len(nodes)
    index = 0
    while index < nodes_count:
        if index in elements:
            index += 1
        else:
            nodes = np.vstack((nodes[:index], nodes[index + 1 :]))
            for i in range(elements.shape[0]):
                for j in range(elements.shape[1]):
                    if elements[i, j] > index:
                        elements[i, j] -= 1
            nodes_count -= 1
    return nodes, elements


@numba.njit
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


@numba.njit
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
        mesh_prop: MeshProperties,
        boundaries_description: BoundariesDescription,
        create_in_subprocess: bool,
    ):
        self.mesh_prop = mesh_prop

        self.initial_nodes: np.ndarray
        self.elements: np.ndarray
        self.edges: np.ndarray

        self.boundaries: Boundaries

        self.base_seed_indices: np.ndarray
        self.closest_seed_index: int

        self.reinitialize_data(mesh_prop, boundaries_description, create_in_subprocess)

    def remesh(self, boundaries_description, create_in_subprocess):
        self.reinitialize_data(self.mesh_prop, boundaries_description, create_in_subprocess)

    def reinitialize_data(
        self,
        mesh_prop: MeshProperties,
        boundaries_description: BoundariesDescription,
        create_in_subprocess,
    ):
        input_nodes, input_elements = mesh_builders.build_mesh(
            mesh_prop=mesh_prop,
            create_in_subprocess=create_in_subprocess,
        )
        unordered_nodes, unordered_elements = remove_unconnected_nodes_numba(
            input_nodes, input_elements
        )
        (
            self.initial_nodes,
            self.elements,
            self.boundaries,
        ) = BoundariesFactory.identify_boundaries_and_reorder_nodes(
            unordered_nodes, unordered_elements, boundaries_description
        )
        self.base_seed_indices, self.closest_seed_index = get_base_seed_indices_numba(
            self.initial_nodes
        )
        edges_matrix = get_edges_matrix(nodes_count=len(self.initial_nodes), elements=self.elements)
        self.edges = get_edges_list_numba(edges_matrix)

    def normalize_shift(self, vectors):
        _ = self
        return vectors - np.mean(vectors, axis=0)

    @property
    def normalized_initial_nodes(self):
        return self.normalize_shift(self.initial_nodes)

    @property
    def input_initial_nodes(self):
        return self.normalized_initial_nodes

    @property
    def boundary_surfaces(self):
        return self.boundaries.boundary_surfaces

    @property
    def contact_boundary(self):
        return self.boundaries.contact_boundary

    @property
    def neumann_boundary(self):
        return self.boundaries.neumann_boundary

    @property
    def dirichlet_boundary(self):
        return self.boundaries.dirichlet_boundary

    @property
    def boundary_internal_indices(self):
        return self.boundaries.boundary_internal_indices

    @property
    def boundary_nodes_count(self):
        return self.boundaries.boundary_nodes_count

    @property
    def contact_nodes_count(self):
        return self.boundaries.contact_nodes_count

    @property
    def dirichlet_nodes_count(self):
        return self.boundaries.dirichlet_nodes_count

    @property
    def neumann_nodes_count(self):
        return self.boundaries.neumann_nodes_count

    @property
    def independent_nodes_count(self):
        return self.nodes_count - self.dirichlet_nodes_count

    @property
    def free_nodes_count(self):
        # TODO: #65 CHECK
        return self.independent_nodes_count - self.contact_nodes_count - self.dirichlet_nodes_count

    @property
    def boundary_indices(self):
        return self.boundaries.boundary_indices

    @property
    def initial_boundary_nodes(self):
        return self.initial_nodes[self.boundary_indices]

    @property
    def contact_indices(self):
        return slice(self.contact_nodes_count)

    @property
    def neumann_indices(self):
        return slice(
            self.contact_nodes_count,
            self.contact_nodes_count + self.neumann_nodes_count,
        )

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
        return self.mesh_prop.dimension

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
