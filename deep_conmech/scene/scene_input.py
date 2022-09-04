import copy
from typing import List

import numba
import numpy as np
import torch

from conmech.helpers import nph
from conmech.properties.body_properties import DynamicBodyProperties
from conmech.properties.mesh_properties import MeshProperties
from conmech.properties.obstacle_properties import ObstacleProperties
from conmech.properties.schedule import Schedule
from deep_conmech.data.data_classes import MeshLayerData, TargetData
from deep_conmech.helpers import thh
from deep_conmech.scene.scene_layers import MeshLayerLinkData
from deep_conmech.scene.scene_randomized import SceneRandomized


@numba.njit
def get_indices_from_graph_sizes_numba(graph_sizes: List[int]):
    index = np.zeros(sum(graph_sizes), dtype=np.int64)
    last = 0
    for size in graph_sizes:
        last += size
        index[last:] += 1
    index = index.reshape(-1, 1)
    return index


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
    forces_from,
    forces_to,
    edges_data_dim,
):
    dimension = initial_nodes_to.shape[1]
    edges_number = edges.shape[0]
    edges_data = np.zeros((edges_number, edges_data_dim))
    for edge_index in range(edges_number):
        j, i = edges[edge_index]

        set_diff_numba(initial_nodes_from, initial_nodes_to, 0, edges_data[edge_index], i, j)
        set_diff_numba(
            displacement_old_from, displacement_old_to, dimension + 1, edges_data[edge_index], i, j
        )
        set_diff_numba(
            velocity_old_from, velocity_old_to, 2 * (dimension + 1), edges_data[edge_index], i, j
        )
        set_diff_numba(forces_from, forces_to, 3 * (dimension + 1), edges_data[edge_index], i, j)
    return edges_data


@numba.njit  # (parallel=True)
def get_multilayer_edges_data_numba(
    edges,
    initial_nodes_sparse,
    initial_nodes_dense,
    displacement_old_sparse,
    displacement_old_dense,
    velocity_old_sparse,
    velocity_old_dense,
    sparse_neighbours,
    edges_data_dim,
):
    dimension = initial_nodes_dense.shape[1]
    size = dimension + 1
    edges_number = edges.shape[0]
    edges_data = np.zeros((edges_number, edges_data_dim))
    for edge_index in range(edges_number):
        j, i = edges[edge_index]
        # n1, n2 = sparse_neighbours[edge_index]

        set_diff_numba(initial_nodes_sparse, initial_nodes_dense, 0, edges_data[edge_index], i, j)
        set_diff_numba(
            displacement_old_sparse, displacement_old_dense, 1 * size, edges_data[edge_index], i, j
        )
        set_diff_numba(
            velocity_old_sparse, velocity_old_dense, 2 * size, edges_data[edge_index], i, j
        )

        # neighbour_data_setter = lambda data_sparse, number: set_diff_numba(
        #     data_sparse,
        #     data_sparse,
        #     number * size,
        #     edges_data[edge_index],
        #     n2,
        #     n1,
        # )

        # neighbour_data_setter(initial_nodes_sparse, 1)
        # neighbour_data_setter(displacement_old_sparse, 2)
        # # neighbour_data_setter(velocity_old_sparse, 5)

    return edges_data


@numba.njit
def get_multilayer_edges_numba(closest_nodes):
    neighbours_count = closest_nodes.shape[1]
    edges_count = closest_nodes.shape[0] * neighbours_count
    edges = np.zeros((edges_count, 2), dtype=np.int64)
    sparse_neighbours = np.zeros((edges_count, neighbours_count - 1), dtype=np.int64)
    # neighbour_ids = np.zeros((edges_count, 1), dtype=np.int64)
    mask = np.ones(neighbours_count, dtype=np.bool8)
    index = 0
    for i, neighbours in enumerate(closest_nodes):
        for neighbour_id, j in enumerate(neighbours):
            edges[index] = [j, i]
            # neighbour_ids[index] = neighbour_id
            mask[neighbour_id] = 0
            sparse_neighbours[index] = neighbours[mask]
            mask[neighbour_id] = 1
            index += 1
    return edges, sparse_neighbours  # , neighbour_ids


class SceneInput(SceneRandomized):
    def __init__(
        self,
        mesh_prop: MeshProperties,
        body_prop: DynamicBodyProperties,
        obstacle_prop: ObstacleProperties,
        schedule: Schedule,
        create_in_subprocess: bool,
    ):
        super().__init__(
            mesh_prop=mesh_prop,
            body_prop=body_prop,
            obstacle_prop=obstacle_prop,
            schedule=schedule,
            create_in_subprocess=create_in_subprocess,
        )

    def prepare_node_data(
        self, data: np.ndarray, layer_number: int, add_norm=False, approximate=True
    ):
        if approximate:
            approximated_data = self.approximate_boundary_or_all_from_base(
                layer_number=layer_number, base_values=data
            )
        else:
            approximated_data = data
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
            forces_from=self.prepare_node_data(
                data=self.input_forces, layer_number=layer_number_from
            ),
            forces_to=self.prepare_node_data(data=self.input_forces, layer_number=layer_number_to),
            edges_data_dim=self.get_edges_data_dim(self.dimension),
        )
        return edges_data

    def get_multilayer_edges_data(self, directional_edges, sparse_neighbours):
        edges_data = get_multilayer_edges_data_numba(
            edges=directional_edges,
            initial_nodes_sparse=self.reduced.input_initial_nodes,
            initial_nodes_dense=self.input_initial_nodes,
            displacement_old_sparse=self.reduced.input_displacement_old,
            displacement_old_dense=self.input_displacement_old,
            velocity_old_sparse=self.reduced.input_velocity_old,
            velocity_old_dense=self.input_velocity_old,
            sparse_neighbours=sparse_neighbours,
            edges_data_dim=self.get_multilayer_edges_data_dim(self.dimension),
        )
        return edges_data

    def get_nodes_data(self, layer_number):
        if layer_number == 1:
            exact_acceleration = self.prepare_node_data(
                layer_number=layer_number,
                data=self.reduced.exact_acceleration,  #########################
                add_norm=True,
                approximate=False,
            )
        # else:
        #     linear_acceleration = self.prepare_node_data(
        #         layer_number=layer_number, data=self.linear_acceleration, add_norm=True
        #     )

        boundary_normals = self.prepare_node_data(
            data=self.get_normalized_boundary_normals(), layer_number=layer_number, add_norm=True
        )
        # boundary_friction = self.prepare_node_data(
        #     data=self.get_friction_input(),
        #     layer_number=layer_number,
        #     add_norm=True,
        # )
        # boundary_normal_response = self.prepare_node_data(
        #     data=self.get_normal_response_input(),
        #     layer_number=layer_number,
        # )
        # boundary_volume = self.prepare_node_data(
        #     data=self.get_surface_per_boundary_node(), layer_number=layer_number
        # )
        if layer_number > 0:
            return np.hstack(
                (
                    exact_acceleration,
                    # linear_acceleration,
                    # input_forces,
                    boundary_normals,
                    # boundary_friction,
                    # boundary_normal_response,
                    # boundary_volume,
                )
            )
        else:
            # input_forces = self.prepare_node_data(
            #     layer_number=layer_number, data=self.input_forces, add_norm=True
            # )
            return np.hstack(
                (
                    # linear_acceleration,
                    # input_forces,
                    boundary_normals,
                    # boundary_normals,
                    # boundary_friction,
                    # boundary_normal_response,
                    # boundary_volume,
                )
            )

    def get_multilayer_edges_with_data(self, link: MeshLayerLinkData):
        closest_nodes = torch.tensor(link.closest_nodes)
        edges_index_np, sparse_neighbours = get_multilayer_edges_numba(link.closest_nodes)
        edges_data = thh.to_torch_set_precision(
            self.get_multilayer_edges_data(
                directional_edges=edges_index_np, sparse_neighbours=sparse_neighbours
            )
        )
        edges_index = thh.get_contiguous_torch(edges_index_np)
        distances_link = link.closest_distances
        closest_nodes_count = link.closest_distances.shape[1]
        distance_norm_index = self.dimension
        distances_edges = (
            edges_data[:, distance_norm_index].numpy().reshape(-1, closest_nodes_count)
        )
        assert np.allclose(distances_link, distances_edges)
        assert np.allclose(
            edges_index_np,
            edges_index.T.numpy(),
        )
        return edges_index, edges_data, closest_nodes

    def get_features_data(self, layer_number: int = 0):
        # edge_index_torch, edge_attr = remove_self_loops(
        #    self.contiguous_edges_torch, self.edges_data_torch
        # )
        # Do not use "face" in any name (reserved in PyG)
        # Do not use "index", "batch" in any name (PyG stacks values to create single graph; batch - adds one, index adds nodes count (?))

        layer_data = self.all_layers[layer_number]
        mesh = layer_data.mesh
        layer_directional_edges = np.vstack((mesh.edges, np.flip(mesh.edges, axis=1)))

        data = MeshLayerData(
            edge_number=torch.tensor([mesh.edges_number]),
            layer_number=torch.tensor([layer_number]),
            pos=thh.to_torch_set_precision(mesh.normalized_initial_nodes),
            x=thh.to_torch_set_precision(self.get_nodes_data(layer_number)),
            # pin_memory=True,
            # num_workers=1
        )

        if layer_number > 0:
            data.layer_nodes_count = torch.tensor([mesh.nodes_count])
            data.down_layer_nodes_count = torch.tensor(
                [self.all_layers[layer_number - 1].mesh.nodes_count]
            )

            (
                data.edge_index_to_down,
                data.edge_attr_to_down,
                data.closest_nodes_to_down,
            ) = self.get_multilayer_edges_with_data(link=layer_data.to_base)

        else:
            data.edge_index = thh.get_contiguous_torch(layer_directional_edges)
            data.edge_attr = thh.to_torch_set_precision(
                self.get_edges_data(
                    layer_directional_edges,
                    layer_number_from=layer_number,
                    layer_number_to=layer_number,
                )
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
        return data

    def get_target_data(self):
        # to_float
        # lhs_sparse = thh.to_double(self.solver_cache.lhs_acceleration_jax).to_sparse()
        # lhs_sparse_copy = copy.deepcopy(lhs_sparse)
        # rhs = thh.to_double(self.get_integrated_forces_column_jax())
        target_data = TargetData(
            a_correction=thh.to_double(self.normalized_a_correction),
            # energy_args=self.get_energy_obstacle_for_jax(None)
            # EnergyObstacleArgumentsTorch(
            #     lhs_values=lhs_sparse_copy.values(),
            #     lhs_indices=lhs_sparse_copy.indices(),
            #     lhs_size=lhs_sparse_copy.size(),
            #     rhs=rhs,
            #     #
            #     # boundary_velocity_old=thh.to_double(self.norm_boundary_velocity_old),
            #     # boundary_normals=thh.to_double(self.get_normalized_boundary_normals()),
            #     # boundary_obstacle_normals=thh.to_double(self.get_norm_boundary_obstacle_normals()),
            #     # penetration=thh.to_double(self.get_penetration_scalar()),
            #     # surface_per_boundary_node=thh.to_double(self.get_surface_per_boundary_node()),
            #     # obstacle_prop=self.obstacle_prop,
            #     # time_step=self.schedule.time_step,
            # ),
            # lhs_values=lhs_sparse.values(),
            # lhs_index=lhs_sparse.indices(),
            # rhs=rhs,
        )

        if hasattr(self, "exact_acceleration"):
            target_data.exact_acceleration = thh.to_double(self.exact_acceleration)
        if hasattr(self, "linear_acceleration"):
            target_data.linear_acceleration = thh.to_double(self.linear_acceleration)
        return target_data

    @staticmethod
    def get_nodes_data_description_sparse(dimension: int):
        desc = []
        for attr in [
            "exact_acceleration",
            # "linear_acceleration",
            # "input_forces",
            "boundary_normals",
            # "boundary_friction",
        ]:
            for i in range(dimension):
                desc.append(f"{attr}_{i}")
            desc.append(f"{attr}_norm")

        # for attr in ["boundary_volume"]:  # "boundary_normal_response", "boundary_volume"]:
        #     desc.append(attr)
        return desc

    @staticmethod
    def get_nodes_data_description_dense(dimension: int):
        desc = []
        for attr in [
            # "linear_acceleration",
            # "input_forces",
            "boundary_normals",
        ]:
            for i in range(dimension):
                desc.append(f"{attr}_{i}")
            desc.append(f"{attr}_norm")

        # for attr in ["boundary_volume"]:
        #     desc.append(attr)
        return desc

    @staticmethod
    def get_nodes_data_down_dim(dimension: int):
        return len(SceneInput.get_nodes_data_description_dense(dimension))

    @staticmethod
    def get_nodes_data_up_dim(dimension: int):
        return len(SceneInput.get_nodes_data_description_sparse(dimension))

    @staticmethod
    def get_edges_data_dim(dimension):
        return len(SceneInput.get_edges_data_description(dimension))

    @staticmethod
    def get_multilayer_edges_data_dim(dimension):
        return (dimension + 1) * 3  # 6

    @staticmethod
    def get_edges_data_description(dim):
        desc = []
        for attr in ["initial_nodes", "displacement_old", "velocity_old", "forces"]:
            for i in range(dim):
                desc.append(f"{attr}_{i}")
            desc.append(f"{attr}_norm")

        return desc
