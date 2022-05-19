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
from conmech.scene.scene import EnergyObstacleArguments, energy_obstacle
from deep_conmech.helpers import thh
from deep_conmech.scene.scene_layers import MeshLayerLinkData, SceneLayers


def energy_normalized_obstacle_correction(cleaned_a, a_correction, args: EnergyObstacleArguments):
    a = cleaned_a if (a_correction is None) else (cleaned_a - a_correction)
    return energy_obstacle(acceleration=a, args=args)


@numba.njit
def set_diff_numba(data_from, data_to, position, row, i, j):
    dimension = data_to.shape[1]
    vector = data_from[j] - data_to[i]
    row[position : position + dimension] = vector
    row[position + dimension] = nph.euclidean_norm_numba(vector)


@numba.njit  # (parallel=True)
def get_edges_data_numba(
    edges,
    initial_nodes_from,
    initial_nodes_to,
    displacement_old_from,
    displacement_old_to,
    velocity_old_from,
    velocity_old_to,
    # forces_from,
    # forces_to,
    edges_data_dim,
):
    dimension = initial_nodes_to.shape[1]
    edges_number = edges.shape[0]
    edges_data = np.zeros((edges_number, edges_data_dim))  # 2 * (dimension + 1))) # 4
    for edge_index in range(edges_number):
        j, i = edges[edge_index]

        set_diff_numba(initial_nodes_from, initial_nodes_to, 0, edges_data[edge_index], i, j)
        set_diff_numba(
            displacement_old_from, displacement_old_to, dimension + 1, edges_data[edge_index], i, j
        )
        set_diff_numba(
            velocity_old_from, velocity_old_to, 2 * (dimension + 1), edges_data[edge_index], i, j
        )
        # set_diff_numba(forces_from, forces_to, 3 * (dimension + 1), edges_data[edge_index], i, j)
    return edges_data


@numba.njit
def get_multilayer_edges_numba(closest_nodes):
    edges_count = closest_nodes.shape[0] * closest_nodes.shape[1]
    edges = np.zeros((edges_count, 2), dtype=np.int64)
    index = 0
    for i, neighbours in enumerate(closest_nodes):
        for j in neighbours:
            edges[index] = [j, i]
            index += 1
    return edges


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


class MeshLayerData(Data):
    # def __init__(self):
    #    super().__init__()

    def __inc__(self, key, value, *args, **kwargs):
        if key == "edge_index_to_base":
            return torch.tensor([[self.layer_nodes_count], [self.base_layer_nodes_count]])
        if key == "edge_index_from_base":
            return torch.tensor([[self.base_layer_nodes_count], [self.layer_nodes_count]])
        if key == "edge_index_to_down":
            return torch.tensor([[self.layer_nodes_count], [self.down_layer_nodes_count]])
        if key == "edge_index_from_down":
            return torch.tensor([[self.down_layer_nodes_count], [self.layer_nodes_count]])
        else:
            return super().__inc__(key, value, *args, **kwargs)


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

    def prepare_node_data(self, data: np.ndarray, layer_number: int, add_norm=False):
        approximated_data = self.approximate_boundary_or_all_from_base(
            layer_number=layer_number, base_values=data
        )
        mesh = self.all_layers[layer_number].mesh
        result = self.complete_mesh_boundary_data_with_zeros(mesh=mesh, data=approximated_data)
        if add_norm:
            result = nph.append_euclidean_norm(result)
        return result

    def get_edges_data(self, directional_edges, layer_number_from: int, layer_number_to: int):
        edges_data = get_edges_data_numba(
            edges=directional_edges,
            initial_nodes_from=self.all_layers[layer_number_from].mesh.input_initial_nodes,
            initial_nodes_to=self.all_layers[layer_number_to].mesh.input_initial_nodes,
            displacement_old_from=self.prepare_node_data(
                data=self.input_displacement_old, layer_number=layer_number_from
            ),
            displacement_old_to=self.prepare_node_data(
                data=self.input_displacement_old, layer_number=layer_number_to
            ),
            velocity_old_from=self.prepare_node_data(
                data=self.input_velocity_old, layer_number=layer_number_from
            ),
            velocity_old_to=self.prepare_node_data(
                data=self.input_velocity_old, layer_number=layer_number_to
            ),
            # forces_from=self.prepare_node_data(
            #    data=self.input_forces, layer_number=layer_number_from
            # ),
            # forces_to=self.prepare_node_data(data=self.input_forces, layer_number=layer_number_to),
            edges_data_dim=self.get_edges_data_dim(self.dimension),
        )
        return thh.to_double(edges_data)

    def get_nodes_data(self, layer_number):
        boundary_penetration = self.prepare_node_data(
            data=self.get_normalized_boundary_penetration(),
            layer_number=layer_number,
            add_norm=True,
        )
        boundary_normals = self.prepare_node_data(
            data=self.get_normalized_boundary_normals(), layer_number=layer_number, add_norm=True
        )
        boundary_v_tangential = self.prepare_node_data(
            data=self.get_normalized_boundary_v_tangential(),
            layer_number=layer_number,
            add_norm=True,
        )
        input_forces = self.prepare_node_data(
            layer_number=layer_number, data=self.input_forces, add_norm=True
        )
        boundary_volume = self.prepare_node_data(
            data=self.get_surface_per_boundary_node(), layer_number=layer_number
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
                # self.get_colliding_nodes_indicator_torch(),
                # self.get_colliding_all_nodes_indicator_torch(),
            )
        )
        return thh.to_double(nodes_data)

    def get_multilayer_edges_with_data(
        self, link: MeshLayerLinkData, layer_number_from: int, layer_number_to: int
    ):
        closest_nodes = torch.tensor(link.closest_nodes)
        closest_weights = thh.set_precision(torch.tensor(link.closest_weights))
        edges_index_np = get_multilayer_edges_numba(link.closest_nodes)
        edges_data = thh.set_precision(
            self.get_edges_data(
                directional_edges=edges_index_np,
                layer_number_from=layer_number_from,
                layer_number_to=layer_number_to,
            )
        )
        edges_index = thh.get_contiguous_torch(edges_index_np)
        distances_link = link.closest_distances
        closest_nodes_count = link.closest_distances.shape[1]
        distance_norm_index = 2
        distances_edges = (
            edges_data[:, distance_norm_index].numpy().reshape(-1, closest_nodes_count)
        )
        assert np.allclose(distances_link, distances_edges)
        return edges_index, edges_data, closest_nodes, closest_weights

    def get_features_data(self, layer_number: int, scene_index: int):
        # exact_normalized_a_torch=None
        # edge_index_torch, edge_attr = remove_self_loops(
        #    self.contiguous_edges_torch, self.edges_data_torch
        # )
        # Do not use "face" in any name (reserved in PyG)
        # Do not use "index", "batch" in any name (PyG stacks values to create single graph; batch - adds one, index adds nodes count (?))

        layer_data = self.all_layers[layer_number]
        mesh = layer_data.mesh
        layer_directional_edges = np.vstack((mesh.edges, np.flip(mesh.edges, axis=1)))

        data = MeshLayerData(
            scene_id=torch.tensor([scene_index]),
            edge_number=torch.tensor([mesh.edges_number]),
            layer_number=torch.tensor([layer_number]),
            node_scene_id=thh.to_long(np.repeat(scene_index, mesh.nodes_count)),
            pos=thh.set_precision(thh.to_double(mesh.normalized_initial_nodes)),
            x=thh.set_precision(self.get_nodes_data(layer_number)),
            edge_index=thh.get_contiguous_torch(layer_directional_edges),
            edge_attr=thh.set_precision(
                self.get_edges_data(
                    layer_directional_edges,
                    layer_number_from=layer_number,
                    layer_number_to=layer_number,
                )
            ),
            # pin_memory=True,
            # num_workers=1
        )

        if layer_number > 0:
            data.layer_nodes_count = torch.tensor([mesh.nodes_count])
            data.down_layer_nodes_count = torch.tensor(
                [self.all_layers[layer_number - 1].mesh.nodes_count]
            )
            data.base_layer_nodes_count = torch.tensor([self.all_layers[0].mesh.nodes_count])
            (
                data.edge_index_to_base,
                data.edge_attr_to_base,
                data.closest_nodes_to_base,
                data.closest_weights_to_base,
            ) = self.get_multilayer_edges_with_data(
                link=layer_data.to_base,
                layer_number_from=layer_number,
                layer_number_to=0,
            )
            (
                data.edge_index_from_base,
                data.edge_attr_from_base,
                data.closest_nodes_from_base,
                data.closest_weights_from_base,
            ) = self.get_multilayer_edges_with_data(
                link=layer_data.from_base,
                layer_number_from=0,
                layer_number_to=layer_number,
            )
            (
                data.edge_index_to_down,
                data.edge_attr_to_down,
                data.closest_nodes_to_down,
                data.closest_weights_to_down,
            ) = self.get_multilayer_edges_with_data(
                link=layer_data.to_down,
                layer_number_from=layer_number,
                layer_number_to=layer_number - 1,
            )
            (
                data.edge_index_from_down,
                data.edge_attr_from_down,
                data.closest_nodes_from_down,
                data.closest_weights_from_down,
            ) = self.get_multilayer_edges_with_data(
                link=layer_data.from_down,
                layer_number_from=layer_number - 1,
                layer_number_to=layer_number,
            )

        return data

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
    def get_nodes_data_dim(dimension: int):
        return len(SceneInput.get_nodes_data_description(dimension))

    @staticmethod
    def get_edges_data_dim(dimension):
        return len(SceneInput.get_edges_data_description(dimension))

    @staticmethod
    def get_edges_data_description(dim):
        desc = []
        for attr in ["initial_nodes", "displacement_old", "velocity_old"]:  # "forces"]:
            for i in range(dim):
                desc.append(f"{attr}_{i}")
            desc.append(f"{attr}_norm")

        return desc
