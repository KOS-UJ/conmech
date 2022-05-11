from dataclasses import dataclass

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
from deep_conmech.scene.scene_layers import SceneLayers


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

    @staticmethod
    def get_edges_data_description(dim):
        desc = []
        for attr in ["initial_nodes", "displacement_old", "velocity_old", "forces"]:
            for i in range(dim):
                desc.append(f"{attr}_{i}")
            desc.append(f"{attr}_norm")

        return desc

    @staticmethod
    def edges_data_dim(dimension):
        return len(SceneInput.get_edges_data_description(dimension))

    def get_edges_data_torch(self, edges):
        edges_data = get_edges_data_numba(
            edges,
            self.normalized_initial_nodes,
            self.input_displacement_old,
            self.input_velocity_old,
            self.input_forces,
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

    def get_nodes_data(self):
        boundary_penetration = self.complete_boundary_data_with_zeros(
            self.get_normalized_boundary_penetration()
        )
        boundary_normals = self.complete_boundary_data_with_zeros(
            self.get_normalized_boundary_normals()
        )
        boundary_v_tangential = self.complete_boundary_data_with_zeros(
            self.get_normalized_boundary_v_tangential()
        )
        boundary_volume = self.complete_boundary_data_with_zeros(
            self.get_surface_per_boundary_node()
        )

        nodes_data = np.hstack(
            (
                nph.append_euclidean_norm(self.input_forces),
                # thh.append_euclidean_norm(self.input_displacement_old_torch),
                # thh.append_euclidean_norm(self.input_velocity_old_torch),
                nph.append_euclidean_norm(boundary_penetration),
                nph.append_euclidean_norm(boundary_normals),
                nph.append_euclidean_norm(boundary_v_tangential),
                boundary_volume,
                # self.get_is_colliding_nodes_torch(),
                # self.get_is_colliding_all_nodes_torch(),
            )
        )
        return thh.to_double(nodes_data)

    def get_data(self, scene_index=None, exact_normalized_a_torch=None):
        # edge_index_torch, edge_attr = remove_self_loops(
        #    self.contiguous_edges_torch, self.edges_data_torch
        # )
        # Do not use "face" in any name (reserved in PyG)
        directional_edges = np.vstack((self.edges, np.flip(self.edges, axis=1)))
        # f"{cmh.get_timestamp(self.config)} - {
        features_data = Data(
            scene_index_str=str(scene_index),  # str; int is changed by PyG
            scene_index=thh.to_long(np.repeat(scene_index, self.nodes_count)),
            pos=thh.set_precision(thh.to_double(self.normalized_initial_nodes)),
            x=thh.set_precision(self.get_nodes_data()),
            edge_index=thh.get_contiguous_torch(directional_edges),
            edge_attr=thh.set_precision(self.get_edges_data_torch(directional_edges)),
            # pin_memory=True,
            # num_workers=1
        )
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
                obstacle_prop=scenarios.default_obstacle_prop,  # TODO: generalize
                time_step=0.01,  # TODO: generalize
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
        return features_data, target_data
