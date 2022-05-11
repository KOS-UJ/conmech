import copy

import numba
import numpy as np

from conmech.helpers import nph
from conmech.properties.body_properties import DynamicBodyProperties
from conmech.properties.mesh_properties import MeshProperties
from conmech.properties.obstacle_properties import ObstacleProperties
from conmech.properties.schedule import Schedule
from deep_conmech.scene.scene_randomized import SceneRandomized


@numba.njit
def get_interlayer_data(old_nodes, new_nodes):
    closest_count = 3
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
        self.all_layers = None
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

            closest_nodes, weighted_closest_distances = get_interlayer_data(
                old_nodes=self.initial_nodes,
                new_nodes=nodes,
            )
            # boundary_surfaces = geom_mesh.cells[0].data.astype("long").copy()
            self.all_layers.append(
                (nodes, elements, boundaries, closest_nodes, weighted_closest_distances)
            )

    def approximate_all(self, layer_number, old_values):
        nodes, edges, boundaries, closest_nodes, weighted_closest_distances = self.all_layers[
            layer_number
        ]
        new_values = np.sum(
            old_values[closest_nodes] * weighted_closest_distances[..., np.newaxis], axis=1
        )
        return new_values
