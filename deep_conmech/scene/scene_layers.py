import copy
from dataclasses import dataclass
from typing import List

import numba
import numpy as np

from conmech.helpers import nph
from conmech.properties.body_properties import DynamicBodyProperties
from conmech.properties.mesh_properties import MeshProperties
from conmech.properties.obstacle_properties import ObstacleProperties
from conmech.properties.schedule import Schedule
from deep_conmech.scene.scene_randomized import SceneRandomized


@numba.njit
def get_interlayer_data(old_nodes, new_nodes, closest_count):
    weighted_closest_distances = np.zeros((len(new_nodes), closest_count))
    closest_nodes = np.zeros((len(new_nodes), closest_count), dtype=np.int64)
    for new_index, new_node in enumerate(new_nodes):
        distances = nph.euclidean_norm_numba(new_node - old_nodes)
        closest_node = distances.argsort()[:closest_count]
        closest_nodes[new_index, :] = closest_node
        closest_distances = distances[closest_node]
        weighted_closest_distances[new_index, :] = closest_distances / np.sum(closest_distances)

        # mean_value = np.sum(old_values[closest_nodes] * weighted_closest_distances, axis=0)
        # new_values[new_index, :] = mean_value
        # new_values[new_index, :] = np.average(
        ##    old_values[closest_nodes], axis=0, weights=distances[closest_nodes]
        # )
    return closest_nodes, weighted_closest_distances


@dataclass
class MeshLayerData:
    nodes: np.ndarray
    elements: np.ndarray
    boundary_nodes: np.ndarray
    closest_nodes: np.ndarray
    weights_closest: np.ndarray
    closest_boundary_nodes: np.ndarray
    weights_closest_boundary: np.ndarray


class SceneLayers(SceneRandomized):
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
        self.all_layers: List[MeshLayerData] = []
        self.set_layers(layers_count=2)

    def set_layers(self, layers_count):
        self.all_layers = []
        is_dirichlet = lambda _: False
        is_contact = lambda _: True
        layer_mesh_prop = copy.deepcopy(self.mesh_prop)
        for _ in range(layers_count):
            layer_mesh_prop.mesh_density = list(
                np.array(layer_mesh_prop.mesh_density, dtype=np.int32) // 2
            )
            (nodes, elements, boundaries) = self.reinitialize_layer(
                layer_mesh_prop, is_dirichlet, is_contact, self.create_in_subprocess
            )
            boundary_nodes = nodes[slice(boundaries.boundary_nodes_count)]

            closest_nodes, weights_closest = get_interlayer_data(
                old_nodes=self.initial_nodes,
                new_nodes=nodes,
                closest_count=self.mesh_prop.dimension + 1
            )
            closest_boundary_nodes, weights_closest_boundary = get_interlayer_data(
                old_nodes=self.boundary_nodes,
                new_nodes=boundary_nodes,
                closest_count=self.mesh_prop.dimension
            )

            mesh_layer_data = MeshLayerData(
                nodes=nodes,
                elements=elements,
                boundary_nodes=boundary_nodes,
                closest_nodes=closest_nodes,
                weights_closest=weights_closest,
                closest_boundary_nodes=closest_boundary_nodes,
                weights_closest_boundary=weights_closest_boundary,
            )
            self.all_layers.append(mesh_layer_data)

    def approximate_internal(self, old_values, closest_nodes, weights_closest):
        return np.sum(
            old_values[closest_nodes] * weights_closest[..., np.newaxis],
            axis=1,
        )

    def approximate_all(self, layer_number, old_values):
        mesh_layer_data = self.all_layers[layer_number]
        return self.approximate_internal(
            old_values=old_values,
            closest_nodes=mesh_layer_data.closest_nodes,
            weights_closest=mesh_layer_data.weights_closest,
        )

    def approximate_boundary(self, layer_number, old_values):
        mesh_layer_data = self.all_layers[layer_number]
        return self.approximate_internal(
            old_values=old_values,
            closest_nodes=mesh_layer_data.closest_boundary_nodes,
            weights_closest=mesh_layer_data.weights_closest_boundary,
        )
