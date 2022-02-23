from typing import Callable

import numpy as np
import numba
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
            False,
            mu_coef,
            la_coef,
            th_coef,
            ze_coef,
            density,
            time_step,
            lambda: self.reorganize_boundaries(is_contact, is_dirichlet),
            False
        )

    def reorganize_boundaries(self, is_contact, is_dirichlet):
        (
            self.boundaries,
            self.initial_points,
            self.cells,
        ) = Boundaries.identify_boundaries_and_reorder_vertices(
            self.initial_points, self.cells, is_contact, is_dirichlet
        )

        self.independent_nodes_count = len(self.initial_points)
        for vertex in reversed(self.initial_points):
            if not is_dirichlet(*vertex):
                break
            self.independent_nodes_count -= 1

        self.contact_num = 0
        for vertex in self.initial_points:
            if not is_contact(*vertex):
                break
            self.contact_num += 1
