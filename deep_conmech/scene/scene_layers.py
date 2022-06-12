import copy
from ctypes import ArgumentError
from dataclasses import dataclass
from typing import List, Optional

import numpy as np

from conmech.mesh.mesh import Mesh
from conmech.properties.body_properties import DynamicBodyProperties
from conmech.properties.mesh_properties import MeshProperties
from conmech.properties.obstacle_properties import ObstacleProperties
from conmech.properties.schedule import Schedule
from deep_conmech.data import interpolation_helpers
from deep_conmech.scene.scene_randomized import SceneRandomized


@dataclass
class MeshLayerLinkData:
    closest_nodes: np.ndarray
    closest_distances: np.ndarray
    closest_weights: Optional[np.ndarray]
    closest_boundary_nodes: np.ndarray
    closest_weights_boundary: np.ndarray
    closest_distances_boundary: np.ndarray


@dataclass
class AllMeshLayerLinkData:
    mesh: Mesh
    to_down: Optional[MeshLayerLinkData]
    from_down: Optional[MeshLayerLinkData]
    from_base: Optional[MeshLayerLinkData]


class SceneLayers(SceneRandomized):
    def __init__(
        self,
        mesh_prop: MeshProperties,
        body_prop: DynamicBodyProperties,
        obstacle_prop: ObstacleProperties,
        schedule: Schedule,
        normalize_by_rotation: bool,
        create_in_subprocess: bool,
        layers_count: int,
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
        self.all_layers: List[AllMeshLayerLinkData] = []
        self.set_layers(layers_count=layers_count)

    def set_layers(self, layers_count):
        self.all_layers = []
        layer_mesh_prop = copy.deepcopy(self.mesh_prop)

        base_mesh_layer_data = AllMeshLayerLinkData(
            mesh=self,
            to_down=None,
            from_down=None,
            from_base=None,
        )
        self.all_layers.append(base_mesh_layer_data)

        dense_mesh = self
        for _ in range(layers_count - 1):
            layer_mesh_prop.mesh_density = list(
                np.array(layer_mesh_prop.mesh_density, dtype=np.int32) // 2
            )

            sparse_mesh = Mesh(
                mesh_prop=layer_mesh_prop,
                is_dirichlet=None,
                is_contact=None,
                create_in_subprocess=self.create_in_subprocess,
            )
            mesh_layer_data = AllMeshLayerLinkData(
                mesh=sparse_mesh,
                to_down=self.get_link(
                    from_mesh=sparse_mesh, to_mesh=dense_mesh, with_weights=False
                ),
                from_down=self.get_link(
                    from_mesh=dense_mesh, to_mesh=sparse_mesh, with_weights=False
                ),
                from_base=self.get_link(from_mesh=self, to_mesh=sparse_mesh, with_weights=True),
            )
            self.all_layers.append(mesh_layer_data)
            dense_mesh = sparse_mesh

    def get_link(self, from_mesh: Mesh, to_mesh: Mesh, with_weights: bool):
        old_nodes = from_mesh.normalized_initial_nodes
        new_nodes = to_mesh.normalized_initial_nodes
        (
            closest_nodes,
            closest_distances,
            closest_weights,
        ) = interpolation_helpers.get_interlayer_data(
            base_nodes=old_nodes,
            interpolated_nodes=new_nodes,
            closest_count=self.mesh_prop.dimension + 1,
            with_weights=with_weights,
        )
        # assert np.allclose(
        #     new_nodes,
        #     nph.elementwise_dot(old_nodes[closest_nodes], closest_weights[..., np.newaxis]),
        # )
        (
            closest_boundary_nodes,
            closest_distances_boundary,
            closest_weights_boundary,
        ) = interpolation_helpers.get_interlayer_data(
            base_nodes=from_mesh.initial_boundary_nodes,
            interpolated_nodes=to_mesh.initial_boundary_nodes,
            closest_count=self.mesh_prop.dimension,
            with_weights=with_weights,
        )
        return MeshLayerLinkData(
            closest_nodes=closest_nodes,
            closest_distances=closest_distances,
            closest_weights=closest_weights,
            closest_boundary_nodes=closest_boundary_nodes,
            closest_distances_boundary=closest_distances_boundary,
            closest_weights_boundary=closest_weights_boundary,
        )

    def approximate_boundary_or_all_from_base(self, layer_number: int, base_values: np.ndarray):
        if layer_number == 0:
            return base_values

        mesh_layer_data = self.all_layers[layer_number]
        link = mesh_layer_data.from_base
        if link is None:
            raise ArgumentError

        if len(base_values) == self.nodes_count:
            closest_nodes = link.closest_nodes
            closest_weights = link.closest_weights

        elif len(base_values) == self.boundary_nodes_count:
            closest_nodes = link.closest_boundary_nodes
            closest_weights = link.closest_weights_boundary
        else:
            raise ArgumentError

        return interpolation_helpers.approximate_internal(
            base_values=base_values, closest_nodes=closest_nodes, closest_weights=closest_weights
        )
