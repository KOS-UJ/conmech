import copy
from ctypes import ArgumentError
from dataclasses import dataclass
from typing import List, Optional

import numba
import numpy as np

from conmech.helpers import cmh, interpolation_helpers, jxh, lnh
from conmech.helpers.config import SimulationConfig
from conmech.mesh.mesh import Mesh
from conmech.properties.body_properties import TimeDependentBodyProperties
from conmech.properties.mesh_properties import MeshProperties
from conmech.properties.obstacle_properties import ObstacleProperties
from conmech.properties.schedule import Schedule
from conmech.scene.scene import Scene
from deep_conmech.training_config import CLOSEST_COUNT


@numba.njit
def get_multilayer_edges_numba(closest_nodes):
    neighbours_count = closest_nodes.shape[1]
    edges_count = closest_nodes.shape[0] * neighbours_count
    edges = np.zeros((edges_count, 2), dtype=np.int64)
    index = 0
    for i, neighbours in enumerate(closest_nodes):
        for _, j in enumerate(neighbours):
            edges[index] = [j, i]
            index += 1
    return edges


@dataclass
class MeshLayerLinkData:
    closest_nodes: np.ndarray
    closest_distances: np.ndarray
    closest_weights: Optional[np.ndarray]
    closest_boundary_nodes: np.ndarray
    closest_weights_boundary: np.ndarray
    closest_distances_boundary: np.ndarray
    edges_index: np.ndarray


@dataclass
class AllMeshLayerLinkData:
    mesh: Mesh
    from_base: Optional[MeshLayerLinkData]
    to_base: Optional[MeshLayerLinkData]


class SceneLayers(Scene):
    def __init__(
        self,
        mesh_prop: MeshProperties,
        body_prop: TimeDependentBodyProperties,
        obstacle_prop: ObstacleProperties,
        schedule: Schedule,
        simulation_config: SimulationConfig,
        create_in_subprocess: bool,
    ):
        super().__init__(
            mesh_prop=mesh_prop,
            body_prop=body_prop,
            obstacle_prop=obstacle_prop,
            schedule=schedule,
            simulation_config=simulation_config,
            create_in_subprocess=create_in_subprocess,
            with_schur=False,
        )
        self.create_in_subprocess = create_in_subprocess
        self.all_layers: List[AllMeshLayerLinkData] = []
        self.set_reduced()

    def project_sparse_nodes(self, from_base, sparse_scene):
        sparse_scene.mesh.initial_nodes = interpolation_helpers.approximate_internal(
            base_values=self.initial_nodes,
            closest_nodes=from_base.closest_nodes,
            closest_weights=from_base.closest_weights,
        )

    def set_reduced(self):
        self.all_layers = []
        layer_mesh_prop = copy.deepcopy(self.mesh_prop)

        base_mesh_layer_data = AllMeshLayerLinkData(
            mesh=self,
            from_base=None,
            to_base=None,
        )
        self.all_layers.append(base_mesh_layer_data)

        layer_mesh_prop.mesh_density = list(
            np.array(layer_mesh_prop.mesh_density, dtype=np.int32)
            // self.simulation_config.mesh_layer_proportion
        )

        reduced_scene = Scene(
            mesh_prop=layer_mesh_prop,
            body_prop=self.body_prop,
            obstacle_prop=self.obstacle_prop,
            schedule=self.schedule,
            simulation_config=self.simulation_config,
            create_in_subprocess=self.create_in_subprocess,
            with_schur=False,
        )
        reduced_scene.lifted_acceleration = np.zeros_like(reduced_scene.initial_nodes)

        if self.simulation_config.mode == "skinning":
            pass
            #     reduced_scene.mesh.initial_nodes += 0.03 * np.array(
            #         jxh.complete_data_with_zeros(
            #             reduced_scene.boundary_normals, reduced_scene.nodes_count
            #         )
            #     )# 0.13 OK 0.12-0.15 # 0.08
            # (
            #     closest_nodes,
            #     closest_distances,
            #     closest_weights,
            # ) = interpolation_helpers.interpolate_nodes(
            #     base_nodes=reduced_scene.initial_nodes,
            #     base_elements=reduced_scene.elements,
            #     query_nodes=self.initial_nodes,
            #     # padding = 0.1
            # )
            # edges_index = get_multilayer_edges_numba(closest_nodes)
            # self.all_layers.append(
            #     AllMeshLayerLinkData(
            #         mesh=reduced_scene,
            #         to_base=MeshLayerLinkData(
            #             closest_nodes=closest_nodes,
            #             closest_distances=closest_distances,
            #             closest_weights=closest_weights,
            #             closest_boundary_nodes=None,
            #             closest_distances_boundary=None,
            #             closest_weights_boundary=None,
            #             edges_index=edges_index,
            #         ),
            #         from_base=self.get_link(
            #             from_mesh=self, to_mesh=reduced_scene, with_weights=True
            #         ),
            #     )
            # )

        from_base = self.get_link(from_mesh=self, to_mesh=reduced_scene, with_weights=True)
        # self.project_sparse_nodes(from_base, reduced_scene) #TODO: reintroduce
        to_base = self.get_link(from_mesh=reduced_scene, to_mesh=self, with_weights=True)  ### False

        mesh_layer_data = AllMeshLayerLinkData(
            mesh=reduced_scene,
            from_base=from_base,
            to_base=to_base,
        )
        self.all_layers.append(mesh_layer_data)

    def get_link(self, from_mesh: Mesh, to_mesh: Mesh, with_weights: bool):
        (
            closest_nodes,
            closest_distances,
            closest_weights,
        ) = interpolation_helpers.interpolate_nodes(  # get_interlayer_data_numba
            base_nodes=from_mesh.initial_nodes,
            base_elements=from_mesh.elements,
            query_nodes=to_mesh.initial_nodes,
            # closest_count=CLOSEST_COUNT,
            # with_weights=with_weights,
            # padding = 0.1
        )
        # (
        #     closest_boundary_nodes,
        #     closest_distances_boundary,
        #     closest_weights_boundary,
        # ) = interpolation_helpers.get_interlayer_data_numba(
        #     base_nodes=from_mesh.initial_boundary_nodes,
        #     base_elements=from_mesh.elements,
        #     interpolated_nodes=to_mesh.initial_boundary_nodes,
        #     closest_count=CLOSEST_BOUNDARY_COUNT,
        #     with_weights=with_weights,
        # )
        edges_index = get_multilayer_edges_numba(closest_nodes)
        return MeshLayerLinkData(
            closest_nodes=closest_nodes,
            closest_distances=closest_distances,
            closest_weights=closest_weights,
            closest_boundary_nodes=None,  # closest_boundary_nodes,
            closest_distances_boundary=None,  # closest_distances_boundary,
            closest_weights_boundary=None,  # closest_weights_boundary,
            edges_index=edges_index,
        )

    @property
    def reduced(self):
        return self.all_layers[1].mesh

    def normalize_and_set_obstacles(
        self,
        obstacles_unnormalized: Optional[np.ndarray],
        all_mesh_prop: Optional[List[MeshProperties]],
    ):
        super().normalize_and_set_obstacles(obstacles_unnormalized, all_mesh_prop)
        self.reduced.normalize_and_set_obstacles(obstacles_unnormalized, all_mesh_prop)

    def lift_data(self, data):
        return self.approximate_boundary_or_all_from_base(layer_number=1, base_values=data)

    def lower_data(self, data):
        return self.approximate_boundary_or_all_to_base(layer_number=1, reduced_values=data)

    def lift_acceleration_from_position(self, acceleration):
        new_displacement = self.to_displacement(acceleration)
        moved_nodes_new = self.initial_nodes + new_displacement

        moved_reduced_nodes_new = self.lift_data(moved_nodes_new)

        new_reduced_displacement = moved_reduced_nodes_new - self.reduced.initial_nodes
        reduced_exact_acceleration = self.reduced.from_displacement(new_reduced_displacement)
        return reduced_exact_acceleration

    def lower_displacement_from_position(self, new_reduced_displacement):
        moved_reduced_nodes_new = self.reduced.initial_nodes + new_reduced_displacement
        moved_nodes_new = self.lower_data(moved_reduced_nodes_new)
        new_displacement = moved_nodes_new - self.initial_nodes
        return new_displacement

    def lower_acceleration_from_position(self, reduced_acceleration):
        new_reduced_displacement = self.reduced.to_displacement(reduced_acceleration)
        new_displacement = self.lower_displacement_from_position(new_reduced_displacement)
        acceleration_from_displacement = self.from_displacement(new_displacement)
        return acceleration_from_displacement

    def prepare(self, inner_forces: np.ndarray):
        super().prepare(inner_forces)
        reduced_inner_forces = self.lift_data(inner_forces)
        self.reduced.prepare(reduced_inner_forces)
        # scene.reduced.prepare(scenario.get_forces_by_function(scene.reduced, current_time))

    def iterate_self(self, acceleration, temperature=None):
        super().iterate_self(acceleration, temperature)
        self.update_reduced()

    def reorient_to_reduced(self, exact_acceleration):
        base_displacement = self.to_displacement(exact_acceleration)
        reduced_displacement_new = self.reduced.to_displacement(self.reduced.exact_acceleration)
        base = self.reduced.get_rotation(reduced_displacement_new)
        position = np.mean(reduced_displacement_new, axis=0)

        new_displacement = self.get_displacement(
            base=base, position=position, base_displacement=base_displacement
        )
        return self.from_displacement(new_displacement)

    def recenter_reduced_mesh(self):
        displacement = self.reduced.get_displacement(base=self.moved_base, position=self.position)
        self.reduced.set_displacement_old(displacement)

    def update_reduced(self):
        # Randomizarion calls this method twice
        if self.reduced.lifted_acceleration is None:
            return
        self.reduced.iterate_self(self.reduced.lifted_acceleration)
        # self.recenter_reduced_mesh()
        # recenter velocity !!!
        self.reduced.lifted_acceleration = None
        return

    def approximate_boundary_or_all_from_base(self, layer_number: int, base_values: np.ndarray):
        if base_values is None or layer_number == 0:
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

    def approximate_boundary_or_all_to_base(self, layer_number: int, reduced_values: np.ndarray):
        if reduced_values is None or layer_number == 0:
            return reduced_values

        mesh_layer_data = self.all_layers[layer_number]
        reduced_scene = mesh_layer_data.mesh
        link = mesh_layer_data.to_base
        if link is None:
            raise ArgumentError

        if len(reduced_values) == reduced_scene.nodes_count:
            closest_nodes = link.closest_nodes
            closest_weights = link.closest_weights

        elif len(reduced_values) == reduced_scene.boundary_nodes_count:
            closest_nodes = link.closest_boundary_nodes
            closest_weights = link.closest_weights_boundary
        else:
            raise ArgumentError

        return interpolation_helpers.approximate_internal(
            base_values=reduced_values, closest_nodes=closest_nodes, closest_weights=closest_weights
        )
