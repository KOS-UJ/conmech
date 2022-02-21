from typing import Callable

import numpy as np
import numba
from conmech.old.boundaries import Boundaries
from deep_conmech.simulator.setting.setting_matrices import SettingMatrices


class MeshFeatures(SettingMatrices):
    def __init__(
        self,
        mesh_size_x,
        mesh_size_y,
        mesh_type,
        corners,
        is_adaptive,
        MU,
        LA,
        TH,
        ZE,
        DENS,
        TIMESTEP,
        is_dirichlet: Callable,
        is_contact: Callable,
    ):
        super().__init__(
            mesh_type=mesh_type,
            mesh_density_x=mesh_size_x,
            mesh_density_y=mesh_size_y,
            scale_x=corners[2],
            scale_y=corners[3],
            is_adaptive=is_adaptive,
            create_in_subprocess=False,
            mu=MU,
            la=LA,
            th=TH,
            ze=ZE,
            density=DENS,
            timestep=TIMESTEP,
            reorganize_boundaries=lambda: self.reorganize_boundaries(
                is_contact, is_dirichlet
            ),
        )

    def reorganize_boundaries(self, is_contact, is_dirichlet):
        (
            self.boundaries,
            self.initial_points,
            self.cells,
        ) = Boundaries.identify_boundaries_and_reorder_vertices(
            self.initial_points, self.cells, is_contact, is_dirichlet
        )

        self.independent_nodes_conunt = len(self.initial_points)
        for vertex in reversed(self.initial_points):
            if not is_dirichlet(*vertex):
                break
            self.independent_nodes_conunt -= 1

        self.contact_num = 0
        for vertex in self.initial_points:
            if not is_contact(*vertex):
                break
            self.contact_num += 1
