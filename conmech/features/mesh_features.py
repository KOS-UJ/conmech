from typing import Callable

from conmech.features.boundaries import Boundaries
from deep_conmech.simulator.setting.setting_matrices import SettingMatrices


class MeshFeatures(SettingMatrices):
    def __init__(
        self,
        mesh_type,
        mesh_density_x,
        mesh_density_y,
        scale_x,
        scale_y,
        is_adaptive,
        mu_coef,
        la_coef,
        th_coef,
        ze_coef,
        density,
        time_step,
        is_dirichlet: Callable,
        is_contact: Callable,
    ):
        super().__init__(
            mesh_type,
            mesh_density_x,
            mesh_density_y,
            scale_x,
            scale_y,
            is_adaptive,
            create_in_subprocess=False,
            mu_coef=mu_coef,
            la_coef=la_coef,
            th_coef=th_coef,
            ze_coef=ze_coef,
            density=density,
            time_step=time_step,
            is_dirichlet=is_dirichlet,
            is_contact=is_contact,
            with_schur_complement_matrices=False,
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
