import numba
import numpy as np
import torch
from torch_geometric.data import Data

from conmech.helpers import nph
from conmech.helpers.config import Config
from conmech.properties.body_properties import DynamicBodyProperties
from conmech.properties.mesh_properties import MeshProperties
from conmech.properties.obstacle_properties import ObstacleProperties
from conmech.properties.schedule import Schedule
from conmech.scenarios import scenarios
from conmech.scene.scene import EnergyObstacleArguments, energy_obstacle
from deep_conmech.helpers import thh
from deep_conmech.scene.scene_torch import SceneTorch


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


class SceneInput(SceneTorch):
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
    def edges_data_dim(dimension):
        return (dimension + 1) * 4

    @staticmethod
    def get_edges_data_description(dim):
        desc = []
        for attr in ["initial_nodes", "displacement_old", "velocity_old", "forces"]:
            for i in range(dim):
                desc.append(f"{attr}_{i}")
            desc.append(f"{attr}_norm")

        return desc

    def get_edges_data_torch(self, edges):
        edges_data = get_edges_data_numba(
            edges,
            self.normalized_initial_nodes,
            self.input_displacement_old,
            self.input_velocity_old,
            self.normalized_inner_forces,
        )
        return thh.to_torch_double(edges_data)

    @staticmethod
    def nodes_data_dim(dimension):
        return (dimension + 1) * 4 + 1  # 19 # 13

    @staticmethod
    def get_nodes_data_description(dim):
        desc = []
        for attr in [
            "forces",
            # "displacement_old",
            # "velocity_old",
            "boundary_penetration",
            "boundary_normals",
            "boundary_v_tangential",
        ]:
            for i in range(dim):
                desc.append(f"{attr}_{i}")
            desc.append(f"{attr}_norm")

        for attr in ["boundary_volume"]:
            desc.append(attr)
        return desc

    def get_nodes_data(self):
        boundary_penetration = self.complete_boundary_data_with_zeros_torch(
            self.get_normalized_boundary_penetration_torch()
        )
        boundary_normals = self.complete_boundary_data_with_zeros_torch(
            self.get_normalized_boundary_normals_torch()
        )
        boundary_v_tangential = self.complete_boundary_data_with_zeros_torch(
            self.get_normalized_boundary_v_tangential_torch()
        )
        boundary_volume = self.complete_boundary_data_with_zeros_torch(
            self.get_surface_per_boundary_node_torch()
        )

        nodes_data = torch.hstack(
            (
                thh.append_euclidean_norm(self.normalized_inner_forces_torch),
                # thh.append_euclidean_norm(self.input_displacement_old_torch),
                # thh.append_euclidean_norm(self.input_velocity_old_torch),
                thh.append_euclidean_norm(boundary_penetration),
                thh.append_euclidean_norm(boundary_normals),
                thh.append_euclidean_norm(boundary_v_tangential),
                boundary_volume,
            )
        )
        return nodes_data

    def get_data(self, scene_index=None, exact_normalized_a_torch=None):
        # edge_index_torch, edge_attr = remove_self_loops(
        #    self.contiguous_edges_torch, self.edges_data_torch
        # )
        # Do not use "face" in any name (reserved in PyG)
        directional_edges = np.vstack((self.edges, np.flip(self.edges, axis=1)))
        # f"{cmh.get_timestamp(self.config)} - {
        features_data = Data(
            scene_index=str(scene_index),  # str; int is changed by PyG
            pos=thh.set_precision(self.normalized_initial_nodes_torch),
            x=thh.set_precision(self.get_nodes_data()),
            edge_index=thh.get_contiguous_torch(directional_edges),
            edge_attr=thh.set_precision(self.get_edges_data_torch(directional_edges)),
            # pin_memory=True,
            # num_workers=1
        )
        target_data = dict(
            a_correction=self.normalized_a_correction_torch,
            args=EnergyObstacleArguments(
                lhs=self.lhs_torch,
                rhs=self.get_normalized_rhs_torch(),
                boundary_velocity_old=self.normalized_boundary_velocity_old_torch,
                boundary_nodes=self.normalized_boundary_nodes_torch,
                boundary_normals=self.get_normalized_boundary_normals_torch(),
                boundary_obstacle_nodes=self.normalized_boundary_obstacle_nodes_torch,
                boundary_obstacle_normals=self.get_normalized_boundary_obstacle_normals_torch(),
                surface_per_boundary_node=self.get_surface_per_boundary_node_torch(),
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
