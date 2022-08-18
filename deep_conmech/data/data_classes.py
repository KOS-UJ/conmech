from dataclasses import dataclass
from typing import List, Optional

import numpy as np
import torch
from torch_geometric.data import Data


class MeshLayerData(Data):
    def __inc__(self, key, value, *args, **kwargs):
        if key == "closest_nodes_to_down":
            return torch.tensor([self.layer_nodes_count])
        if key == "edge_index_to_down":
            return torch.tensor([[self.layer_nodes_count], [self.down_layer_nodes_count]])
        else:
            return super().__inc__(key, value, *args, **kwargs)


"""

    lhs_acceleration_jax: np.ndarray
    rhs_acceleration: np.ndarray
    boundary_velocity_old: np.ndarray
    boundary_normals: np.ndarray
    boundary_obstacle_normals: np.ndarray
    penetration: np.ndarray
    surface_per_boundary_node: np.ndarray
    body_prop: np.ndarray
    obstacle_prop: np.ndarray
    time_step: float
    element_initial_volume: np.ndarray
    dx_big_jax: np.ndarray
    base_displacement: np.ndarray
    base_velocity: np.ndarray
    base_energy_displacement: np.ndarray
    base_energy_velocity: np.ndarray

"""


@dataclass
class EnergyObstacleArgumentsTorch:
    lhs_values: torch.Tensor = None
    lhs_indices: torch.Tensor = None
    lhs_size: torch.Size = None
    rhs: torch.Tensor = None
    # lhs_sparse: Optional[torch.Tensor] = None
    # boundary_velocity_old: torch.Tensor
    # boundary_normals: torch.Tensor
    # boundary_obstacle_normals: torch.Tensor
    # penetration: torch.Tensor
    # surface_per_boundary_node: torch.Tensor
    # obstacle_prop: ObstacleProperties
    # time_step: float

    # def to(self, device: torch.device, non_blocking: bool = False):
    #     self_vars = vars(self)
    #     for (key, value) in self_vars.items():
    #         if hasattr(value, "to"):
    #             self_vars[key] = value.to(device, non_blocking=non_blocking)
    #     return self


class TargetData(Data):
    def __init__(
        self,
        a_correction: torch.Tensor,
        # energy_args: EnergyObstacleArgumentsTorch,
        # lhs_values: torch.Tensor,
        # lhs_index: torch.Tensor,
        # rhs: torch.Tensor,
    ):
        super().__init__()
        self.a_correction = a_correction
        # self.energy_args = energy_args

        # self.lhs_values = lhs_values
        # self.lhs_index = lhs_index
        # self.rhs = rhs

    def __inc__(self, key, value, *args, **kwargs):
        if key == "lhs_index":
            size = self.a_correction.numel()
            return torch.tensor([[size], [size]])
        else:
            return super().__inc__(key, value, *args, **kwargs)


@dataclass
class GraphData:
    layer_list: List[MeshLayerData]
    target_data: TargetData
    scene: float
