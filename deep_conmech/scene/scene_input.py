from typing import List

import numba
import numpy as np
import torch

from conmech.helpers import nph
from conmech.mesh.mesh import Mesh, mesh_normalization_decorator
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


def get_multilayer_edges_data_new(
    edges,
    initial_nodes_sparse,
    initial_nodes_dense,
    displacement_old_sparse,
    displacement_old_dense,
    velocity_old_sparse,
    velocity_old_dense,
    forces_sparse,
    forces_dense,
):
    def get_column(data_sparse, data_dense):
        column = data_sparse[edges[:, 0]] - data_dense[edges[:, 1]]
        return np.hstack((column, nph.euclidean_norm(column, keepdims=True)))

    return np.hstack(
        (
            get_column(initial_nodes_sparse, initial_nodes_dense),
            # get_column(
            #     displacement_old_sparse,
            #     displacement_old_dense,
            # ),
            # get_column(velocity_old_sparse, velocity_old_dense),
            get_column(forces_sparse, forces_dense),
        )
    )


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

    @mesh_normalization_decorator
    def prepare_node_data(self, scene, data: np.ndarray, add_norm=False):
        # if approximate:
        #     approximated_data = self.approximate_boundary_or_all_from_base(
        #         layer_number=layer_number, base_values=data
        #     )
        # else:
        approximated_data = data
        # mesh = self.all_layers[layer_number].mesh
        result = self.complete_mesh_boundary_data_with_zeros(mesh=scene, data=approximated_data)
        if add_norm:
            result = nph.append_euclidean_norm(result)
        return result

    @mesh_normalization_decorator
    def get_edges_data(self, directional_edges, reduced=False):
        scene = self.reduced if reduced else self

        def get_column(data):
            column = data[directional_edges[:, 1]] - data[directional_edges[:, 0]]
            return np.hstack((column, nph.euclidean_norm(column, keepdims=True)))

        initial_nodes = scene.input_initial_nodes
        forces = self.prepare_node_data(scene=scene, data=scene.input_forces)

        if reduced:
            displacement_old = self.prepare_node_data(
                scene=scene, data=scene.input_displacement_old
            )
            velocity_old = self.prepare_node_data(scene=scene, data=scene.input_velocity_old)
            return np.hstack(
                (
                    get_column(initial_nodes),
                    get_column(displacement_old),
                    get_column(velocity_old),
                    get_column(forces),
                )
            )
        return np.hstack(
            (
                get_column(initial_nodes),
                get_column(forces),
            )
        )

    @mesh_normalization_decorator
    def get_multilayer_edges_data(self, directional_edges):
        # displacement_old_sparse = self.reduced.input_displacement_old
        # displacement_old_dense = self.input_displacement_old
        # velocity_old_sparse = self.reduced.input_velocity_old
        # velocity_old_dense = self.input_velocity_old

        def get_column(data_sparse, data_dense):
            column = data_sparse[directional_edges[:, 0]] - data_dense[directional_edges[:, 1]]
            return np.hstack((column, nph.euclidean_norm(column, keepdims=True)))

        return np.hstack(
            (
                get_column(self.reduced.input_initial_nodes, self.input_initial_nodes),
                # get_column(
                #     displacement_old_sparse,
                #     displacement_old_dense,
                # ),
                # get_column(velocity_old_sparse, velocity_old_dense),
                get_column(self.reduced.input_forces, self.input_forces),
            )
        )

    @mesh_normalization_decorator
    def get_nodes_data(self, reduced):
        scene = self.reduced if reduced else self

        boundary_normals = self.prepare_node_data(
            scene=scene,
            data=scene.get_normalized_boundary_normals(),
            add_norm=True,
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
        if reduced:
            acceleration = self.prepare_node_data(
                scene=scene,
                data=scene.normalized_exact_acceleration,  # scene.exact_acceleration
                add_norm=True,
            )
            new_displacement = self.prepare_node_data(
                scene=scene,
                data=scene.norm_exact_new_displacement,
                add_norm=True,
            )
            return np.hstack(
                (
                    acceleration,
                    new_displacement,
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
                # TODO: Add previous accelerations
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

    @mesh_normalization_decorator
    def get_multilayer_edges_with_data(self, link: MeshLayerLinkData):
        closest_nodes = torch.tensor(link.closest_nodes)
        edges_index_np = get_multilayer_edges_numba(link.closest_nodes)
        edges_data = thh.to_torch_set_precision(
            self.get_multilayer_edges_data(directional_edges=edges_index_np)
        )
        edges_index = thh.get_contiguous_torch(edges_index_np)
        # distances_link = link.closest_distances
        # closest_nodes_count = link.closest_distances.shape[1]
        # distance_norm_index = self.dimension
        # distances_edges = (
        #     edges_data[:, distance_norm_index].numpy().reshape(-1, closest_nodes_count)
        # )
        # assert np.allclose(distances_link, distances_edges)
        # assert np.allclose(
        #     edges_index_np,
        #     edges_index.T.numpy(),
        # )
        return edges_index, edges_data, closest_nodes

    @mesh_normalization_decorator
    def get_features_data(self, layer_number: int = 0):
        # edge_index_torch, edge_attr = remove_self_loops(
        #    self.contiguous_edges_torch, self.edges_data_torch
        # )
        # Do not use "face" in any name (reserved in PyG)
        # Do not use "index", "batch" in any name (PyG stacks values to create single graph; batch - adds one, index adds nodes count (?))
        reduced = layer_number > 0
        layer_data = self.all_layers[layer_number]
        mesh = layer_data.mesh
        layer_directional_edges = np.vstack((mesh.edges, np.flip(mesh.edges, axis=1)))

        data = MeshLayerData(
            edge_number=torch.tensor([mesh.edges_number]),
            layer_number=torch.tensor([layer_number]),
            pos=thh.to_torch_set_precision(mesh.normalized_initial_nodes),
            x=thh.to_torch_set_precision(self.get_nodes_data(reduced=reduced)),
            # pin_memory=True,
            # num_workers=1
        )

        if reduced:
            data.layer_nodes_count = torch.tensor([mesh.nodes_count])
            data.down_layer_nodes_count = torch.tensor(
                [self.all_layers[layer_number - 1].mesh.nodes_count]
            )

            (
                data.edge_index_to_down,
                data.edge_attr_to_down,
                data.closest_nodes_to_down,
            ) = self.get_multilayer_edges_with_data(link=layer_data.to_base)

        data.edge_index = thh.get_contiguous_torch(layer_directional_edges)
        data.edge_attr = thh.to_torch_set_precision(
            self.get_edges_data(layer_directional_edges, reduced=reduced)
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

    @mesh_normalization_decorator
    def get_target_data(self):
        target_data = TargetData()
        target_data.normalized_exact_acceleration = thh.to_double(
            self.normalized_exact_acceleration
        )
        target_data.normalized_new_displacement = thh.to_double(self.norm_exact_new_displacement)
        target_data.reduced_norm_lifted_new_displacement = thh.to_double(
            self.reduced.norm_lifted_new_displacement
        )
        target_data.reduced_normalized_lifted_acceleration = thh.to_double(
            self.reduced.normalized_lifted_acceleration
        )
        return target_data

    @staticmethod
    def get_nodes_data_description_sparse(dimension: int):
        desc = []
        for attr in [
            "exact_acceleration",
            "scaled_new_displacement",
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
    def get_sparse_edges_data_dim(dimension):
        return (dimension + 1) * 4

    @staticmethod
    def get_dense_edges_data_dim(dimension):
        return (dimension + 1) * 2
        # return len(SceneInput.get_edges_data_description(dimension))

    @staticmethod
    def get_multilayer_edges_data_dim(dimension):
        return (dimension + 1) * 2

    @staticmethod
    def get_edges_data_description(dim):
        desc = []
        for attr in ["initial_nodes", "displacement_old"]:  # , "velocity_old", "forces"]:
            for i in range(dim):
                desc.append(f"{attr}_{i}")
            desc.append(f"{attr}_norm")

        return desc
