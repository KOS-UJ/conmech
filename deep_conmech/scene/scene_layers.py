import copy

import numba
import numpy as np
import torch
from torch_geometric.data import Data

from conmech.helpers import nph
from conmech.properties.body_properties import DynamicBodyProperties
from conmech.properties.mesh_properties import MeshProperties
from conmech.properties.obstacle_properties import ObstacleProperties
from conmech.properties.schedule import Schedule
from conmech.scenarios import scenarios
from conmech.scene.scene import EnergyObstacleArguments, energy_obstacle
from deep_conmech.helpers import thh
from deep_conmech.scene.scene_torch import SceneTorch


class SceneLayers(SceneTorch):
    def __init__(
        self,
        mesh_prop: MeshProperties,
        body_prop: DynamicBodyProperties,
        obstacle_prop: ObstacleProperties,
        schedule: Schedule,
        normalize_by_rotation: bool,
        create_in_subprocess: bool,
        with_schur: bool = True,
    ):
        super().__init__(
            mesh_prop=mesh_prop,
            body_prop=body_prop,
            obstacle_prop=obstacle_prop,
            schedule=schedule,
            normalize_by_rotation=normalize_by_rotation,
            create_in_subprocess=create_in_subprocess,
            with_schur=with_schur,
        )
        self.create_in_subprocess = create_in_subprocess
        self.all_layers = None
        self.set_layers()

    def set_layers(self):
        LAYERS = 2

        self.all_layers = []
        is_dirichlet = lambda _: False
        is_contact = lambda _: True
        layer_mesh_prop = copy.deepcopy(self.mesh_prop)
        for _ in range(LAYERS):
            layer_mesh_prop.mesh_density = list(
                np.array(layer_mesh_prop.mesh_density, dtype=np.int32) // 2
            )
            (nodes, elements, boundaries) = self.reinitialize_layer(
                layer_mesh_prop, is_dirichlet, is_contact, self.create_in_subprocess
            )
            # boundary_surfaces = geom_mesh.cells[0].data.astype("long").copy()
            self.all_layers.append((nodes, elements, boundaries))
