import numba
import numpy as np

from conmech.helpers import nph
from conmech.mesh import mesh_builders
from conmech.mesh.boundaries_description import BoundariesDescription
from conmech.mesh.boundaries_factory import BoundariesFactory
from conmech.mesh.boundaries import Boundaries
from conmech.properties.mesh_description import MeshDescription
from conmech.mesh.zoo.raw_mesh import RawMesh


def get_edges_list(elements: np.ndarray) -> np.ndarray:
    """
    Pairs `(i, j)`, `i < j`, of vertices sharing an element, sorted.
    """
    elements = np.asarray(elements, dtype=np.int64)
    vertices = elements.shape[1]
    pairs = np.concatenate(
        [elements[:, (i, j)] for i in range(vertices) for j in range(i + 1, vertices)]
    )
    pairs = np.sort(pairs, axis=1)
    return np.unique(pairs, axis=0)


@numba.njit
def remove_unconnected_nodes_numba(nodes, elements):
    """
    Drop the nodes no element refers to, keeping the order of the rest.
    """
    nodes_count = len(nodes)
    used = np.zeros(nodes_count, dtype=np.bool_)
    for i in range(elements.shape[0]):
        for j in range(elements.shape[1]):
            used[elements[i, j]] = True

    old_to_new = np.zeros(nodes_count, dtype=elements.dtype)
    kept = 0
    for index in range(nodes_count):
        if used[index]:
            old_to_new[index] = kept
            kept += 1

    remaining_nodes = np.empty((kept, nodes.shape[1]), dtype=nodes.dtype)
    for index in range(nodes_count):
        if used[index]:
            remaining_nodes[old_to_new[index]] = nodes[index]

    for i in range(elements.shape[0]):
        for j in range(elements.shape[1]):
            elements[i, j] = old_to_new[elements[i, j]]

    return remaining_nodes, elements


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


class Mesh(RawMesh):
    def __init__(
        self,
        mesh_descr: MeshDescription,
        boundaries_description: BoundariesDescription,
    ):
        self.edges: np.ndarray

        self.boundaries: Boundaries

        input_nodes, input_elements = mesh_builders.build_mesh(mesh_descr=mesh_descr)
        super().__init__(input_nodes, input_elements)
        unordered_nodes, unordered_elements = remove_unconnected_nodes_numba(
            input_nodes, input_elements
        )
        (
            self.nodes,
            self.elements,
            self.boundaries,
        ) = BoundariesFactory.identify_boundaries_and_reorder_nodes(
            unordered_nodes, unordered_elements, boundaries_description
        )
        self.edges = get_edges_list(self.elements)

    @property
    def dimension(self):
        return self.nodes.shape[1]

    @property
    def scale(self):
        return np.max(self.nodes, axis=0) - np.min(self.nodes, axis=0)

    def normalize_shift(self, vectors):
        _ = self
        return vectors - np.mean(vectors, axis=0)

    @property
    def normalized_initial_nodes(self):
        return self.normalize_shift(self.nodes)

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
        return self.nodes[self.boundary_indices]

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
    def nodes_count(self):
        return len(self.nodes)

    @property
    def boundary_surfaces_count(self):
        return len(self.boundary_surfaces)

    @property
    def inner_nodes_count(self):
        return self.nodes_count - self.boundary_nodes_count

    @property
    def edges_number(self):
        return len(self.edges)
