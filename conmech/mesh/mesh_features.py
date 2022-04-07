from typing import Callable

from conmech.properties.body_properties import StaticBodyProperties
from conmech.mesh.mesh_properties import MeshProperties
from conmech.properties.schedule import Schedule
from conmech.mesh.boundaries import Boundaries
from deep_conmech.simulator.dynamics.dynamics import Dynamics


class MeshFeatures(Dynamics):
    def __init__(
            self,
            mesh_data: MeshProperties,
            body_prop: StaticBodyProperties,
            schedule: Schedule,
            normalize_by_rotation: bool,
            is_dirichlet: Callable,
            is_contact: Callable,
    ):
        super().__init__(
            mesh_data=mesh_data,
            body_prop=body_prop,
            schedule=schedule,
            normalize_by_rotation=normalize_by_rotation,
            is_dirichlet=is_dirichlet,
            is_contact=is_contact,
            with_schur_complement_matrices=False,
            create_in_subprocess=False,
        )

    def reorganize_boundaries(self, unordered_nodes, unordered_elements, is_dirichlet, is_contact):
        (
            self.boundaries,
            self.initial_nodes,
            self.elements,
        ) = Boundaries.identify_boundaries_and_reorder_vertices(
            unordered_nodes, unordered_elements, is_contact, is_dirichlet
        )

        dirichlet_nodes_count = 0
        for vertex in reversed(self.initial_nodes):
            if not is_dirichlet(vertex):
                break
            dirichlet_nodes_count += 1

        contact_nodes_count = 0
        for vertex in self.initial_nodes:
            if not is_contact(vertex):
                break
            contact_nodes_count += 1
        
        self.dirichlet_nodes_count = dirichlet_nodes_count
        self.contact_nodes_count = contact_nodes_count
