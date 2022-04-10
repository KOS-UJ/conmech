import numba
import numpy as np
import torch
from torch_geometric.data import Data

from conmech.helpers.config import Config
from conmech.properties.body_properties import DynamicBodyProperties
from conmech.properties.mesh_properties import MeshProperties
from conmech.properties.obstacle_properties import ObstacleProperties
from conmech.properties.schedule import Schedule
from deep_conmech.graph.setting.setting_torch import SettingTorch
from deep_conmech.helpers import thh
from deep_conmech.simulator.setting.setting_obstacles import energy_obstacle


def energy_normalized_obstacle_correction(
    cleaned_a,
    a_correction,
    C,
    E,
    boundary_v_old,
    boundary_nodes,
    boundary_normals,
    boundary_obstacle_nodes,
    boundary_obstacle_normals,
    surface_per_boundary_node,
    obstacle_prop,
    time_step,
):
    a = cleaned_a if (a_correction is None) else (cleaned_a - a_correction)
    return energy_obstacle(
        a=a,
        C=C,
        E=E,
        boundary_v_old=boundary_v_old,
        boundary_nodes=boundary_nodes,
        boundary_normals=boundary_normals,
        boundary_obstacle_nodes=boundary_obstacle_nodes,
        boundary_obstacle_normals=boundary_obstacle_normals,
        surface_per_boundary_node=surface_per_boundary_node,
        obstacle_prop=obstacle_prop,
        time_step=time_step,
    )


@numba.njit
def set_diff(data, position, row, i, j):
    vector = data[j] - data[i]
    row[position : position + 2] = vector
    row[position + 2] = np.linalg.norm(vector)


@numba.njit  # (parallel=True)
def get_edges_data(
    edges,
    initial_nodes,
    u_old,
    v_old,
    forces,
):
    edges_number = edges.shape[0]
    edges_data = np.zeros((edges_number, 12))
    for e in range(edges_number):
        i = edges[e, 0]
        j = edges[e, 1]

        set_diff(initial_nodes, 0, edges_data[e], i, j)
        set_diff(u_old, 3, edges_data[e], i, j)
        set_diff(v_old, 6, edges_data[e], i, j)
        set_diff(forces, 9, edges_data[e], i, j)
    return edges_data


def energy_obstacle_nvt(
    boundary_a,
    C_boundary,
    E_boundary,
    boundary_v_old,
    boundary_nodes,
    boundary_normals,
    boundary_obstacle_nodes,
    boundary_obstacle_normals,
    surface_per_boundary_node,
    config,
):  # np via torch
    value_torch = energy_normalized_obstacle_correction(
        thh.to_torch_double(boundary_a).to(thh.device(config)),
        None,
        thh.to_torch_double(C_boundary).to(thh.device(config)),
        thh.to_torch_double(E_boundary).to(thh.device(config)),
        thh.to_torch_double(boundary_v_old).to(thh.device(config)),
        thh.to_torch_double(boundary_nodes).to(thh.device(config)),
        thh.to_torch_long(boundary_normals).to(thh.device(config)),
        thh.to_torch_double(boundary_obstacle_nodes).to(thh.device(config)),
        thh.to_torch_double(boundary_obstacle_normals).to(thh.device(config)),
        thh.to_torch_double(surface_per_boundary_node).to(thh.device(config)),
    )
    value = thh.to_np_double(value_torch)
    return value  # .item()


class SettingInput(SettingTorch):
    def __init__(
        self,
        mesh_data: MeshProperties,
        body_prop: DynamicBodyProperties,
        obstacle_prop: ObstacleProperties,
        schedule: Schedule,
        config: Config,
        create_in_subprocess: bool,
    ):
        super().__init__(
            mesh_data=mesh_data,
            body_prop=body_prop,
            obstacle_prop=obstacle_prop,
            schedule=schedule,
            config=config,
            create_in_subprocess=create_in_subprocess,
        )

    @staticmethod
    def edges_data_dim():
        return 12

    @staticmethod
    def get_edges_data_description(dim):
        desc = []
        for attr in ["initial_nodes", "u_old", "v_old", "forces"]:
            for i in range(dim):
                desc.append(f"{attr}_{i}")
            desc.append(f"{attr}_norm")

        return desc

    def get_edges_data_torch(self, edges):
        edges_data = get_edges_data(
            edges,
            self.normalized_initial_nodes,
            self.input_u_old,
            self.input_v_old,
            self.input_forces,
        )
        return thh.to_torch_double(edges_data)

    @staticmethod
    def nodes_data_dim():
        return 13

    @staticmethod
    def get_nodes_data_description(dim):
        desc = []
        for attr in [
            "forces",
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
            self.normalized_boundary_penetration_torch
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
                thh.append_euclidean_norm(self.input_forces_torch),
                # thh.append_euclidean_norm(self.input_u_old_torch),
                # thh.append_euclidean_norm(self.input_v_old_torch)
                thh.append_euclidean_norm(boundary_penetration),
                thh.append_euclidean_norm(boundary_normals),
                thh.append_euclidean_norm(boundary_v_tangential),
                boundary_volume,
            )
        )
        return nodes_data

    def get_data(self, setting_index=None, exact_normalized_a_torch=None):
        # edge_index_torch, edge_attr = remove_self_loops(
        #    self.contiguous_edges_torch, self.edges_data_torch
        # )
        # Do not use "face" in name (probably reserved in PyG)
        directional_edges = np.vstack((self.edges, np.flip(self.edges, axis=1)))
        data = Data(
            pos=thh.set_precision(self.normalized_initial_nodes_torch),
            x=thh.set_precision(self.get_nodes_data()),
            edge_index=thh.get_contiguous_torch(directional_edges),
            edge_attr=thh.set_precision(self.get_edges_data_torch(directional_edges)),
            setting_index=setting_index,
            normalized_a_correction=self.normalized_a_correction_torch,
            reshaped_C=self.C_torch.reshape(-1, 1),
            normalized_E=self.get_normalized_E_torch(),
            exact_normalized_a=exact_normalized_a_torch,
            normalized_boundary_v_old=self.normalized_boundary_v_old_torch,
            normalized_boundary_nodes=self.normalized_boundary_nodes_torch,
            normalized_boundary_normals=self.get_normalized_boundary_normals_torch(),
            normalized_boundary_obstacle_nodes=self.normalized_boundary_obstacle_nodes_torch,
            normalized_boundary_obstacle_normals=self.normalized_boundary_obstacle_normals_torch,
            surf_per_boundary_node=self.get_surface_per_boundary_node_torch(),
            boundary_nodes_count=self.boundary_nodes_count_torch,
            # pin_memory=True,
            # num_workers=1
        )
        """
        transform = T.Compose(
            [
                T.TargetIndegree(norm=False),
                T.Cartesian(norm=False),
                T.Polar(norm=False),
            ]  # add custom for multiple 'pos' types
        )  # T.OneHotDegree(),
        transform(data)
        """
        return data

    def normalized_energy_obstacle_nvt(self, normalized_boundary_a_vector):
        normalized_boundary_normals = self.get_normalized_boundary_normals()
        surface_per_boundary_node = self.get_surface_per_boundary_node()
        return energy_obstacle_nvt(
            nph.unstack(normalized_boundary_a_vector, self.dim),
            self.lhs_boundary,
            self.normalized_E_boundary,
            self.normalized_boundary_v_old,
            self.normalized_boundary_nodes,
            normalized_boundary_normals,
            self.normalized_boundary_obstacle_nodes,
            self.normalized_boundary_obstacle_normals,
            surface_per_boundary_node,
        )
