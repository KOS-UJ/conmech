from typing import Callable

from conmech.dataclass.body_properties import StaticBodyProperties
from conmech.dataclass.mesh_data import MeshData
from conmech.dataclass.schedule import Schedule
from conmech.features.boundaries import Boundaries
from deep_conmech.simulator.dynamics.dynamics import Dynamics


class MeshFeatures(Dynamics):
    def __init__(
            self,
            mesh_data: MeshData,
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

    def reorganize_boundaries(self, unordered_nodes, unordered_elements):
        (
            self.boundaries,
            self.initial_nodes,
            self.elements,
        ) = Boundaries.identify_boundaries_and_reorder_vertices(
            unordered_nodes, unordered_elements, self.is_contact, self.is_dirichlet
        )

        self.independent_nodes_count = len(self.initial_nodes)
        for vertex in reversed(self.initial_nodes):
            if not self.is_dirichlet(vertex):
                break
            self.independent_nodes_count -= 1

        self.contact_count = 0
        for vertex in self.initial_nodes:
            if not self.is_contact(vertex):
                break
            self.contact_count += 1

        self.dirichlet_count = len(self.initial_nodes) - self.independent_nodes_count
