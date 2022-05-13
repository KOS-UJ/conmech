from dataclasses import dataclass
from typing import List, Optional

import numba
import numpy as np
import torch
from torch_geometric.data import Data

from conmech.helpers import nph
from conmech.mesh.mesh import Mesh
from conmech.properties.body_properties import DynamicBodyProperties
from conmech.properties.mesh_properties import MeshProperties
from conmech.properties.obstacle_properties import ObstacleProperties
from conmech.properties.schedule import Schedule
from conmech.scene.scene import EnergyObstacleArguments, energy_obstacle
from deep_conmech.helpers import thh
from deep_conmech.scene.scene_layers import MeshLayerLinkData, SceneLayers


def energy_normalized_obstacle_correction(cleaned_a, a_correction, args: EnergyObstacleArguments):
    a = cleaned_a if (a_correction is None) else (cleaned_a - a_correction)
    return energy_obstacle(acceleration=a, args=args)


@numba.njit
def set_diff_numba(data, position, row, i, j):
    dimension = data.shape[1]
    vector = data[j] - data[i]
    row[position : position + dimension] = vector
    row[position + dimension] = nph.euclidean_norm_numba(vector)


@numba.njit  # (parallel=True)
def get_edges_data_numba(
    edges,
    initial_nodes,
    displacement_old,
    velocity_old,
    forces,
):
    dimension = initial_nodes.shape[1]
    edges_number = edges.shape[0]
    edges_data = np.zeros((edges_number, 4 * (dimension + 1)))
    for e in range(edges_number):
        i = edges[e, 0]
        j = edges[e, 1]

        set_diff_numba(initial_nodes, 0, edges_data[e], i, j)
        set_diff_numba(displacement_old, dimension + 1, edges_data[e], i, j)
        set_diff_numba(velocity_old, 2 * (dimension + 1), edges_data[e], i, j)
        set_diff_numba(forces, 3 * (dimension + 1), edges_data[e], i, j)
    return edges_data


@dataclass
class EnergyObstacleArgumentsTorch:
    lhs: torch.Tensor
    rhs: torch.Tensor
    boundary_velocity_old: torch.Tensor
    boundary_nodes: torch.Tensor
    boundary_normals: torch.Tensor
    boundary_obstacle_nodes: torch.Tensor
    boundary_obstacle_normals: torch.Tensor
    surface_per_boundary_node: torch.Tensor
    obstacle_prop: ObstacleProperties
    time_step: float


class SceneInput(SceneLayers):
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
            layers_count=layers_count,
            with_schur=with_schur,
        )

    @staticmethod
    def get_edges_data_description(dim):
        desc = []
        for attr in ["initial_nodes", "displacement_old", "velocity_old", "forces"]:
            for i in range(dim):
                desc.append(f"{attr}_{i}")
            desc.append(f"{attr}_norm")

        return desc

    def prepare_data(self, data: np.ndarray, layer_number: int, add_norm=True):
        new_data = self.approximate_boundary_or_all_from_base(
            layer_number=layer_number, base_values=data
        )
        mesh = self.all_layers[layer_number].mesh
        result = self.complete_boundary_data_with_zeros_layer(mesh=mesh, data=new_data)
        if add_norm:
            result = nph.append_euclidean_norm(result)
        return result

    @staticmethod
    def edges_data_dim(dimension):
        return len(SceneInput.get_edges_data_description(dimension))

    def get_edges_data_torch(self, directional_edges, layer_number: int):
        edges_data = get_edges_data_numba(
            directional_edges,
            self.prepare_data(
                data=self.input_initial_nodes, layer_number=layer_number, add_norm=False
            ),
            self.prepare_data(
                data=self.input_displacement_old, layer_number=layer_number, add_norm=False
            ),
            self.prepare_data(
                data=self.input_velocity_old, layer_number=layer_number, add_norm=False
            ),
            self.prepare_data(data=self.input_forces, layer_number=layer_number, add_norm=False),
        )
        return thh.to_double(edges_data)

    @staticmethod
    def get_nodes_data_description(dimension: int):
        desc = []
        for attr in [
            "forces",
            # "displacement_old",
            # "velocity_old",
            "boundary_penetration",
            "boundary_normals",
            "boundary_v_tangential",
        ]:
            for i in range(dimension):
                desc.append(f"{attr}_{i}")
            desc.append(f"{attr}_norm")

        for attr in ["boundary_volume"]:  # , "is_colliding_nodes", "is_colliding_all_nodes"]:
            desc.append(attr)
        return desc

    @staticmethod
    def nodes_data_dim(dimension: int):
        return len(SceneInput.get_nodes_data_description(dimension))

    def get_nodes_data(self, layer_number):
        boundary_penetration = self.prepare_data(
            data=self.get_normalized_boundary_penetration(), layer_number=layer_number
        )
        boundary_normals = self.prepare_data(
            data=self.get_normalized_boundary_normals(), layer_number=layer_number
        )
        boundary_v_tangential = self.prepare_data(
            data=self.get_normalized_boundary_v_tangential(), layer_number=layer_number
        )
        input_forces = self.prepare_data(layer_number=layer_number, data=self.input_forces)
        boundary_volume = self.prepare_data(
            data=self.get_surface_per_boundary_node(), layer_number=layer_number, add_norm=False
        )

        nodes_data = np.hstack(
            (
                input_forces,
                # self.input_displacement_old_torch,
                # self.input_velocity_old_torch,
                boundary_penetration,
                boundary_normals,
                boundary_v_tangential,
                boundary_volume,
                # self.get_is_colliding_nodes_torch(),
                # self.get_is_colliding_all_nodes_torch(),
            )
        )
        return thh.to_double(nodes_data)

    def complete_boundary_data_with_zeros_layer(self, mesh: Mesh, data):
        # return np.resize(data, (self.nodes_count, data.shape[1]))
        if len(data) == mesh.nodes_count:
            return data
        completed_data = np.zeros((mesh.nodes_count, data.shape[1]), dtype=data.dtype)
        completed_data[mesh.boundary_indices] = data
        return completed_data

    def get_features_data(self, layer_number: int, scene_index: int):
        # exact_normalized_a_torch=None
        # edge_index_torch, edge_attr = remove_self_loops(
        #    self.contiguous_edges_torch, self.edges_data_torch
        # )
        # Do not use "face" in any name (reserved in PyG)
        # Do not use "index", "batch" in any name (PyG stacks values to create single graph; batch - adds one, index adds nodes count (?))

        mesh_layer_data = self.all_layers[layer_number]
        mesh = mesh_layer_data.mesh
        directional_edges = np.vstack((mesh.edges, np.flip(mesh.edges, axis=1)))

        def get_closest_nodes(link: Optional[MeshLayerLinkData]):
            return None if link is None else torch.tensor(link.closest_nodes)

        def get_closest_weights(link: Optional[MeshLayerLinkData]):
            return None if link is None else thh.set_precision(torch.tensor(link.weights_closest))

        features_data = Data(
            scene_id=torch.tensor([scene_index]),
            layer_number=torch.tensor([layer_number]),
            node_scene_id=thh.to_long(np.repeat(scene_index, mesh.nodes_count)),
            pos=thh.set_precision(thh.to_double(mesh.normalized_initial_nodes)),
            x=thh.set_precision(self.get_nodes_data(layer_number)),
            edge_index=thh.get_contiguous_torch(directional_edges),
            edge_attr=thh.set_precision(self.get_edges_data_torch(directional_edges, layer_number)),
            ###
            closest_nodes_from_down=get_closest_nodes(mesh_layer_data.link_from_down),
            closest_weights_from_down=get_closest_weights(mesh_layer_data.link_from_down),
            #
            closest_nodes_from_base=get_closest_nodes(mesh_layer_data.link_from_base),
            closest_weights_from_base=get_closest_weights(mesh_layer_data.link_from_base),
            #
            closest_nodes_to_down=get_closest_nodes(mesh_layer_data.link_to_down),
            closest_weights_to_down=get_closest_weights(mesh_layer_data.link_to_down),
            #
            closest_nodes_to_base=get_closest_nodes(mesh_layer_data.link_to_base),
            closest_weights_to_base=get_closest_weights(mesh_layer_data.link_to_base),
            # pin_memory=True,
            # num_workers=1
        )
        return features_data

    def get_target_data(self):
        target_data = dict(
            a_correction=thh.to_double(self.normalized_a_correction),
            args=EnergyObstacleArgumentsTorch(
                lhs=thh.to_double(self.solver_cache.lhs),
                rhs=thh.to_double(self.get_normalized_rhs_np()),
                boundary_velocity_old=thh.to_double(self.norm_boundary_velocity_old),
                boundary_nodes=thh.to_double(self.normalized_boundary_nodes),
                boundary_normals=thh.to_double(self.get_normalized_boundary_normals()),
                boundary_obstacle_nodes=thh.to_double(self.norm_boundary_obstacle_nodes),
                boundary_obstacle_normals=thh.to_double(self.get_norm_boundary_obstacle_normals()),
                surface_per_boundary_node=thh.to_double(self.get_surface_per_boundary_node()),
                obstacle_prop=self.obstacle_prop,
                time_step=self.schedule.time_step,
            ),
        )
        _ = """
        transform = T.Compose(
            [
                T.TargetIndegree(norm=False),
                T.Cartesian(norm=False),
                T.Polar(norm=False),
            ]  # add custom for multiple 'pos' types
        )  # T.OneHotDegree(),
        transform(data)
        """
        return target_data
