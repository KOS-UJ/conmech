from typing import List

import jax
import jax.numpy as jnp
import numba
import numpy as np
import torch

from conmech.helpers import jxh, nph
from conmech.helpers.config import SimulationConfig
from conmech.properties.body_properties import TimeDependentBodyProperties
from conmech.properties.mesh_properties import MeshProperties
from conmech.properties.obstacle_properties import ObstacleProperties
from conmech.properties.schedule import Schedule
from conmech.state.body_position import mesh_normalization_decorator
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


def prepare_node_data(data: np.ndarray, nodes_count, add_norm=False):
    approximated_data = jnp.array(data)
    result = jxh.complete_data_with_zeros(data=approximated_data, nodes_count=nodes_count)
    if add_norm:
        result = jxh.append_euclidean_norm(result)
    return result


def get_edges_column(data_from, data_to, directional_edges):
    print("get_edges_column")
    column = data_to[directional_edges[:, 1]] - data_from[directional_edges[:, 0]]

    return jnp.hstack((column, nph.euclidean_norm(column, keepdims=True)))


class SceneInput(SceneRandomized):
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
        )

    @mesh_normalization_decorator
    def get_edges_data(self, directional_edges, reduced=False):
        scene = self.reduced if reduced else self

        def get_column(data):
            data_jax = jnp.array(data)
            return jax.jit(get_edges_column)(
                data_from=data_jax,
                data_to=data_jax,
                directional_edges=directional_edges,
            )

        # TODO: Add historical data
        if reduced:
            return jnp.hstack(
                (
                    get_column(scene.input_initial_nodes),  # cached
                    get_column(scene.input_displacement_old),
                    get_column(scene.input_velocity_old),
                    get_column(scene.input_forces),
                )
            )
        return jnp.hstack(
            (
                get_column(scene.input_initial_nodes),
                get_column(scene.input_forces),
            )
        )

    @mesh_normalization_decorator
    def get_multilayer_edges_data(self, directional_edges):
        # displacement_old_sparse = self.reduced.input_displacement_old
        # displacement_old_dense = self.input_displacement_old
        # velocity_old_sparse = self.reduced.input_velocity_old
        # velocity_old_dense = self.input_velocity_old

        def get_column(data_sparse, data_dense):
            return jax.jit(get_edges_column)(
                data_from=jnp.array(data_sparse),
                data_to=jnp.array(data_dense),
                directional_edges=directional_edges,
            )

        return np.hstack(
            (
                get_column(self.reduced.input_initial_nodes, self.input_initial_nodes),  # cached
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

        def prepare_nodes(data):
            return jax.jit(prepare_node_data, static_argnames=["add_norm", "nodes_count"])(
                data=data,
                add_norm=True,
                nodes_count=scene.nodes_count,
            )

        boundary_normals = prepare_nodes(scene.get_normalized_boundary_normals_jax())

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
        input_forces = prepare_nodes(scene.input_forces)
        if reduced:
            new_displacement = prepare_nodes(scene.norm_exact_new_displacement)
            # new_displacement = prepare_nodes(
            #     scene.to_normalized_displacement_rotated_displaced(scene.lifted_acceleration)
            # )
            return jnp.hstack(
                (
                    new_displacement,
                    # linear_acceleration,
                    boundary_normals,
                    # boundary_friction,
                    # boundary_normal_response,
                    # boundary_volume,
                    input_forces,
                )
            )
        else:
            new_lowered_displacement = self.lower_displacement_from_position(
                self.reduced.norm_exact_new_displacement
            )
            # new_randomized_displacement = self.to_normalized_displacement(0 * self.exact_acceleration) #use old acceleration

            # if self.simulation_config.mode != "net": # TODO: Clean
            #     def get_random(scale):
            #         return nph.generate_normal(
            #             rows=self.nodes_count,
            #             columns=self.dimension,
            #             sigma=scale,
            #         )

            #     randomization = get_random(scale= (scene.time_step**2))
            #     new_randomized_displacement += randomization

            return jnp.hstack(
                # TODO: Add previous accelerations
                (
                    # prepare_nodes(new_randomized_displacement),
                    new_lowered_displacement,
                    # linear_acceleration,
                    boundary_normals,
                    # boundary_normals,
                    # boundary_friction,
                    # boundary_normal_response,
                    # boundary_volume,
                    input_forces,
                )
            )

    @mesh_normalization_decorator
    def get_multilayer_edges_with_data(self, link: MeshLayerLinkData):
        closest_nodes = torch.tensor(link.closest_nodes)
        edges_data = thh.to_torch_set_precision(
            self.get_multilayer_edges_data(directional_edges=link.edges_index)
        )
        edges_index = thh.get_contiguous_torch(link.edges_index)
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
    def get_features_data(self, layer_number: int = 0, to_cpu=False):
        # edge_index_torch, edge_attr = remove_self_loops(
        #    self.contiguous_edges_torch, self.edges_data_torch
        # )
        # Do not use "face" in any name (reserved in PyG)
        # Do not use "index", "batch" in any name (PyG stacks values to create single graph; batch - adds one, index adds nodes count (?))
        reduced = layer_number > 0
        layer_data = self.all_layers[layer_number]
        scene = layer_data.mesh

        data = MeshLayerData(
            edge_number=torch.tensor([scene.edges_number]),
            layer_number=torch.tensor([layer_number]),
            pos=thh.to_torch_set_precision(scene.normalized_initial_nodes),
            x=thh.convert_jax_to_tensor_set_precision(self.get_nodes_data(reduced=reduced)),
            # pin_memory=True,
            # num_workers=1
        )

        if reduced:
            data.layer_nodes_count = torch.tensor([scene.nodes_count])
            data.down_layer_nodes_count = torch.tensor(
                [self.all_layers[layer_number - 1].mesh.nodes_count]
            )

            (
                data.edge_index_to_down,
                data.edge_attr_to_down,
                data.closest_nodes_to_down,
            ) = self.get_multilayer_edges_with_data(link=layer_data.to_base)

        data.edge_index = thh.get_contiguous_torch(scene.mesh.directional_edges)
        data.edge_attr = thh.convert_jax_to_tensor_set_precision(
            self.get_edges_data(scene.mesh.directional_edges, reduced=reduced)
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
        if to_cpu:
            data.x = data.x.cpu()
            data.edge_attr = data.edge_attr.cpu()

        return data

    @mesh_normalization_decorator
    def get_target_data(self, to_cpu=False):
        _ = to_cpu
        target_data = TargetData()
        new_norm_lowered_displacement = self.lower_displacement_from_position(
            self.reduced.norm_exact_new_displacement
        )
        new_lowered_displacement = self.lower_displacement_from_position(
            self.reduced.to_displacement(self.reduced.exact_acceleration)
        )
        lifted_new_displacement = self.to_displacement(self.lifted_acceleration)

        # target_data.normalized_new_displacement = thh.to_double(self.norm_exact_new_displacement)
        target_data.normalized_new_displacement = thh.to_double(self.norm_lifted_new_displacement)
        # target_data.normalized_new_displacement = thh.to_double(
        #     self.to_normalized_displacement_rotated_displaced(self.lifted_acceleration)
        # )

        target_data.reduced_norm_lifted_new_displacement = thh.to_double(
            self.reduced.norm_lifted_new_displacement
        )
        target_data.last_displacement_step = thh.to_double(self.get_last_displacement_step())
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
