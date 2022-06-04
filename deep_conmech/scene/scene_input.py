from dataclasses import dataclass
from typing import Optional

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
from conmech.scene.body_forces import energy
from conmech.scene.scene import get_boundary_integral
from deep_conmech.graph.loss_raport import LossRaport
from deep_conmech.helpers import thh
from deep_conmech.scene.scene_layers import MeshLayerLinkData, SceneLayers


class MeshLayerData(Data):
    def __inc__(self, key, value, *args, **kwargs):
        if key == "closest_nodes_to_down":
            return torch.tensor([self.layer_nodes_count])
        if key == "closest_nodes_from_down":
            return torch.tensor([self.down_layer_nodes_count])
        if key == "edge_index_to_down":
            return torch.tensor([[self.layer_nodes_count], [self.down_layer_nodes_count]])
        if key == "edge_index_from_down":
            return torch.tensor([[self.down_layer_nodes_count], [self.layer_nodes_count]])
        else:
            return super().__inc__(key, value, *args, **kwargs)


@dataclass
class EnergyObstacleArgumentsTorch:
    lhs: torch.Tensor
    rhs: torch.Tensor
    boundary_velocity_old: torch.Tensor
    boundary_normals: torch.Tensor
    boundary_obstacle_normals: torch.Tensor
    penetration: torch.Tensor
    surface_per_boundary_node: torch.Tensor
    obstacle_prop: ObstacleProperties
    time_step: float

    def to(self, device: torch.device, non_blocking: bool = False):
        self_vars = vars(self)
        for (key, value) in self_vars.items():
            if hasattr(value, "to"):
                self_vars[key] = value.to(device, non_blocking=non_blocking)
        return self


@dataclass
class TargetData:
    a_correction: torch.Tensor
    energy_args: EnergyObstacleArgumentsTorch

    def to(self, device, non_blocking: bool = False):
        self.a_correction = self.a_correction.to(device, non_blocking=non_blocking)
        self.energy_args = self.energy_args.to(device, non_blocking=non_blocking)
        return self


def clean_acceleration(cleaned_a, a_correction):
    return cleaned_a if (a_correction is None) else (cleaned_a - a_correction)


def get_mean_loss(acceleration, forces, mass_density, boundary_integral):
    # F = m * a
    return (boundary_integral == 0) * (
        torch.norm(torch.mean(forces, axis=0) - torch.mean(mass_density * acceleration, axis=0))
        ** 2
    )


def loss_normalized_obstacle_correction(
    cleaned_a: torch.Tensor,
    a_correction: torch.Tensor,
    forces: torch.Tensor,
    energy_args: EnergyObstacleArgumentsTorch,
    exact_a: Optional[torch.Tensor],
):

    acceleration = clean_acceleration(cleaned_a=cleaned_a, a_correction=a_correction)

    inner_energy = energy(acceleration, energy_args.lhs, energy_args.rhs)
    boundary_integral = get_boundary_integral(acceleration=acceleration, args=energy_args)
    loss_energy = inner_energy + boundary_integral

    loss_mean = get_mean_loss(
        acceleration=acceleration,
        forces=forces,
        mass_density=scenarios.default_body_prop.mass_density,
        boundary_integral=boundary_integral,
    )
    main_loss = loss_energy  # loss_mean + 0.01 * loss_energy

    loss_raport = LossRaport(
        main=main_loss.item(),
        inner_energy=inner_energy.item(),
        energy=loss_energy.item(),
        boundary_integral=boundary_integral.item(),
        mean=loss_mean.item(),
        _count=1,
    )

    if exact_a is not None:
        loss_raport.rmse = thh.to_np_double(thh.rmse_torch(cleaned_a, exact_a))
        exact_energy = energy(exact_a, energy_args.lhs, energy_args.rhs) + get_boundary_integral(
            acceleration=exact_a, args=energy_args
        )
        loss_raport.relative_energy = thh.to_np_double(
            (loss_raport.energy - exact_energy) / torch.abs(exact_energy)
        )
    # en = point_energy(acceleration=acceleration, node_features=node_features, dimension=dimension)
    return main_loss, loss_raport


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
            #     data=self.input_forces, layer_number=layer_number_from
            # ),
            # forces_to=self.prepare_node_data(data=self.input_forces, layer_number=layer_number_to),
            edges_data_dim=self.get_edges_data_dim(self.dimension),
        )
        return edges_data

    def get_nodes_data(self, layer_number):
        input_forces = self.prepare_node_data(
            layer_number=layer_number, data=self.input_forces, add_norm=True
        )
        boundary_normals = self.prepare_node_data(
            data=self.get_normalized_boundary_normals(), layer_number=layer_number, add_norm=True
        )
        boundary_friction = self.prepare_node_data(
            data=self.get_friction_input(),
            layer_number=layer_number,
            add_norm=True,
        )
        boundary_normal_response = self.prepare_node_data(
            data=self.get_normal_response_input(),
            layer_number=layer_number,
        )
        boundary_volume = self.prepare_node_data(
            data=self.get_surface_per_boundary_node(), layer_number=layer_number
        )
        nodes_data = np.hstack(
            (
                input_forces,
                boundary_normals,
                boundary_friction,
                boundary_normal_response,
                boundary_volume,
            )
        )
        return nodes_data

    def get_multilayer_edges_with_data(
        self, link: MeshLayerLinkData, layer_number_from: int, layer_number_to: int
    ):
        closest_nodes = torch.tensor(link.closest_nodes)
        edges_index_np = get_multilayer_edges_numba(link.closest_nodes)
        edges_data = thh.to_torch_set_precision(
            self.get_edges_data(
                directional_edges=edges_index_np,
                layer_number_from=layer_number_from,
                layer_number_to=layer_number_to,
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

    def get_features_data(self, layer_number: int):
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
            edge_number=torch.tensor([mesh.edges_number]),
            layer_number=torch.tensor([layer_number]),
            pos=thh.to_torch_set_precision(mesh.normalized_initial_nodes),
            x=thh.to_torch_set_precision(self.get_nodes_data(layer_number)),
            edge_index=thh.get_contiguous_torch(layer_directional_edges),
            edge_attr=thh.to_torch_set_precision(
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

            (
                data.edge_index_to_down,
                data.edge_attr_to_down,
                data.closest_nodes_to_down,
            ) = self.get_multilayer_edges_with_data(
                link=layer_data.to_down,
                layer_number_from=layer_number,
                layer_number_to=layer_number - 1,
            )
            (
                data.edge_index_from_down,
                data.edge_attr_from_down,
                data.closest_nodes_from_down,
            ) = self.get_multilayer_edges_with_data(
                link=layer_data.from_down,
                layer_number_from=layer_number - 1,
                layer_number_to=layer_number,
            )

        return data

    def get_target_data(self):
        lhs_torch = thh.to_double(self.solver_cache.lhs).to_sparse()
        target_data = dict(  # TargetData
            a_correction=thh.to_double(self.normalized_a_correction),
            args=EnergyObstacleArgumentsTorch(
                lhs=lhs_torch,
                rhs=thh.to_double(self.get_normalized_rhs_np()),
                boundary_velocity_old=thh.to_double(self.norm_boundary_velocity_old),
                boundary_normals=thh.to_double(self.get_normalized_boundary_normals()),
                boundary_obstacle_normals=thh.to_double(self.get_norm_boundary_obstacle_normals()),
                penetration=thh.to_double(self.get_penetration_scalar()),
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
            "input_forces",
            "boundary_normals",
            "boundary_friction",
        ]:
            for i in range(dimension):
                desc.append(f"{attr}_{i}")
            desc.append(f"{attr}_norm")

        for attr in ["boundary_damping", "boundary_volume"]:
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
        for attr in ["initial_nodes", "displacement_old", "velocity_old"]:  # , "forces"]:
            for i in range(dim):
                desc.append(f"{attr}_{i}")
            desc.append(f"{attr}_norm")

        return desc
