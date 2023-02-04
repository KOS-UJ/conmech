import numba
import numpy as np

from conmech.helpers import cmh
from conmech.mesh import mesh_builders
from conmech.mesh.boundaries import Boundaries
from conmech.mesh.boundaries_description import BoundariesDescription
from conmech.mesh.boundaries_factory import BoundariesFactory
from conmech.properties.mesh_properties import MeshProperties


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


# pylint: disable=R0904
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
        self.directional_edges = self.get_directional_edges()

    def get_directional_edges(self):
        size = self.elements.shape[1]
        directional_edges = np.array(
            list(
                {(e[i], e[j]) for i, j in np.ndindex((size, size)) if j != i for e in self.elements}
            ),
            dtype=np.int64,
        )  # j > i - non-directional edges
        return directional_edges

    @property
    def edges_number(self):
        if self.directional_edges is None:
            raise AttributeError()
        return len(self.directional_edges) // 2

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
    def centered_initial_nodes(self):
        return self.initial_nodes - np.mean(self.initial_nodes, axis=0)
