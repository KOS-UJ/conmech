from typing import Callable

import numba
import numpy as np

from conmech.helpers import cmh
from conmech.helpers.config import (
    NORMALIZE,
    USE_CONSTANT_CONTACT_INTEGRAL,
    USE_GREEN_STRAIN,
    USE_NONCONVEX_FRICTION_LAW,
)
from conmech.mesh import mesh_builders
from conmech.mesh.boundaries import Boundaries
from conmech.mesh.boundaries_description import BoundariesDescription
from conmech.mesh.boundaries_factory import BoundariesFactory
from conmech.properties.mesh_properties import MeshProperties


@numba.njit
def get_edges_matrix(nodes_count: int, elements: np.ndarray):
    edges_matrix = np.zeros((nodes_count, nodes_count), dtype=np.int32)
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
        dtype=np.int64,
    )
    return edges


@numba.njit
def remove_unconnected_nodes_numba(nodes, elements):
    nodes_count = len(nodes)
    present_nodes = np.zeros(nodes_count, dtype=numba.boolean)
    for element in elements:
        for node in element:
            present_nodes[node] = True

    index = 0
    removed = 0
    while index < nodes_count:
        if present_nodes[index + removed]:
            index += 1
        else:
            # print("Removing node...")
            nodes = np.vstack((nodes[:index], nodes[index + 1 :]))
            for i in range(elements.shape[0]):
                for j in range(elements.shape[1]):
                    if elements[i, j] > index:
                        elements[i, j] -= 1
            removed += 1
            nodes_count -= 1
    return nodes, elements


def mesh_normalization_decorator(func: Callable):
    def inner(self, *args, **kwargs):
        saved_normalize = self.normalize
        self.mesh_prop.normalize = True
        if hasattr(self, "reduced"):
            self.reduced.mesh_prop.normalize = True
        returned_value = func(self, *args, **kwargs)
        self.mesh_prop.normalize = saved_normalize
        if hasattr(self, "reduced"):
            self.reduced.mesh_prop.normalize = saved_normalize
        return returned_value

    return inner


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

        def fun_data():
            self.reinitialize_data(mesh_prop, boundaries_description, create_in_subprocess)

        cmh.profile(fun_data, baypass=True)

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
            unordered_nodes=unordered_nodes,
            unordered_elements=unordered_elements,
            boundaries_description=boundaries_description,
        )
        edges_matrix = get_edges_matrix(nodes_count=self.nodes_count, elements=self.elements)
        self.edges = get_edges_list_numba(edges_matrix)

    def _normalize_shift(self, vectors):
        _ = self
        if not self.normalize:
            return vectors
        return vectors - np.mean(vectors, axis=0)

    @property
    def normalize(self):
        if hasattr(self.mesh_prop, "normalize"):
            return self.mesh_prop.normalize
        return NORMALIZE

    @property
    def use_green_strain(self):
        if hasattr(self.mesh_prop, "use_green_strain"):
            return self.mesh_prop.use_green_strain
        return USE_GREEN_STRAIN

    @property
    def use_nonconvex_friction_law(self):
        if hasattr(self.mesh_prop, "use_nonconvex_friction_law"):
            return self.mesh_prop.use_nonconvex_friction_law
        return USE_NONCONVEX_FRICTION_LAW

    @property
    def use_constant_contact_integral(self):
        if hasattr(self.mesh_prop, "use_constant_contact_integral"):
            return self.mesh_prop.use_constant_contact_integral
        return USE_CONSTANT_CONTACT_INTEGRAL

    @property
    def normalized_initial_nodes(self):
        return self._normalize_shift(self.initial_nodes)

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
    def elements_count(self):
        return len(self.elements)

    @property
    def boundary_surfaces_count(self):
        return len(self.boundary_surfaces)

    @property
    def inner_nodes_count(self):
        return self.nodes_count - self.boundary_nodes_count

    @property
    def edges_number(self):
        return len(self.edges)

    @property
    def centered_initial_nodes(self):
        return self.initial_nodes - np.mean(self.initial_nodes, axis=0)
