import copy
from ctypes import ArgumentError
from dataclasses import dataclass
from typing import List, Optional

import numba
import numpy as np
import torch

from conmech.helpers import nph
from conmech.mesh.mesh import Mesh
from conmech.properties.body_properties import DynamicBodyProperties
from conmech.properties.mesh_properties import MeshProperties
from conmech.properties.obstacle_properties import ObstacleProperties
from conmech.properties.schedule import Schedule
from deep_conmech.scene.scene_randomized import SceneRandomized


@dataclass
class MeshLayerLinkData:
    closest_nodes: np.ndarray
    weights_closest: np.ndarray
    closest_boundary_nodes: np.ndarray
    weights_closest_boundary: np.ndarray


@dataclass
class MeshLayerData:
    mesh: Mesh
    link_to_down: Optional[MeshLayerLinkData]
    link_from_down: Optional[MeshLayerLinkData]
    link_to_base: Optional[MeshLayerLinkData]
    link_from_base: Optional[MeshLayerLinkData]


@numba.njit
def get_interlayer_data(old_nodes, new_nodes, closest_count):
    closest_weights = np.zeros((len(new_nodes), closest_count))
    closest_nodes = np.zeros((len(new_nodes), closest_count), dtype=np.int64)
    for new_index, new_node in enumerate(new_nodes):
        distances = nph.euclidean_norm_numba(new_node - old_nodes)
        closest_node_list = distances.argsort()[:closest_count]
        base_nodes = old_nodes[closest_node_list]
        closest_weight_list = new_node @ np.linalg.pinv(base_nodes)  # Moore-Penrose pseudo-inverse
        # (np.linalg.pinv(base_nodes).T @ new_node.reshape(-1, 1)).reshape(-1)

        closest_nodes[new_index, :] = closest_node_list
        closest_weights[new_index, :] = closest_weight_list

    return closest_nodes, closest_weights


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
        self.all_layers: List[MeshLayerData] = []
        self.set_layers(layers_count=layers_count)

    def set_layers(self, layers_count):
        self.all_layers = []
        is_dirichlet = lambda _: False
        is_contact = lambda _: True
        layer_mesh_prop = copy.deepcopy(self.mesh_prop)

        base_mesh_layer_data = MeshLayerData(
            mesh=self,
            link_to_down=None,
            link_from_down=None,
            link_to_base=None,
            link_from_base=None,
        )
        self.all_layers.append(base_mesh_layer_data)

        dense_mesh = self
        for _ in range(layers_count - 1):
            layer_mesh_prop.mesh_density = list(
                np.array(layer_mesh_prop.mesh_density, dtype=np.int32) // 2
            )

            sparse_mesh = Mesh(
                mesh_prop=layer_mesh_prop,
                is_dirichlet=is_dirichlet,
                is_contact=is_contact,
                create_in_subprocess=self.create_in_subprocess,
            )
            mesh_layer_data = MeshLayerData(
                mesh=sparse_mesh,
                link_to_down=self.get_link(from_mesh=sparse_mesh, to_mesh=dense_mesh),
                link_from_down=self.get_link(from_mesh=dense_mesh, to_mesh=sparse_mesh),
                link_to_base=self.get_link(from_mesh=sparse_mesh, to_mesh=self),
                link_from_base=self.get_link(from_mesh=self, to_mesh=sparse_mesh),
            )
            self.all_layers.append(mesh_layer_data)
            dense_mesh = sparse_mesh

    def get_link(self, from_mesh: Mesh, to_mesh: Mesh):
        closest_nodes, weights_closest = get_interlayer_data(
            old_nodes=from_mesh.initial_nodes,
            new_nodes=to_mesh.initial_nodes,
            closest_count=self.mesh_prop.dimension + 1,
        )
        closest_boundary_nodes, weights_closest_boundary = get_interlayer_data(
            old_nodes=from_mesh.initial_boundary_nodes,
            new_nodes=to_mesh.initial_boundary_nodes,
            closest_count=self.mesh_prop.dimension,
        )

        return MeshLayerLinkData(
            closest_nodes=closest_nodes,
            weights_closest=weights_closest,
            closest_boundary_nodes=closest_boundary_nodes,
            weights_closest_boundary=weights_closest_boundary,
        )

    @staticmethod
    def approximate_internal(from_values, closest_nodes, closest_weights):
        return (
            from_values[closest_nodes] * closest_weights.reshape(*closest_weights.shape, 1)
        ).sum(axis=1)

    def approximate_boundary_or_all_from_base(self, layer_number: int, base_values: np.ndarray):
        if layer_number == 0:
            return base_values

        mesh_layer_data = self.all_layers[layer_number]
        link = mesh_layer_data.link_from_base
        if link is None:
            raise ArgumentError

        if len(base_values) == self.nodes_count:
            closest_nodes = link.closest_nodes
            weights_closest = link.weights_closest

        elif len(base_values) == self.boundary_nodes_count:
            closest_nodes = link.closest_boundary_nodes
            weights_closest = link.weights_closest_boundary
        else:
            raise ArgumentError

        return SceneLayers.approximate_internal(
            from_values=base_values, closest_nodes=closest_nodes, closest_weights=weights_closest
        )
