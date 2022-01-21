import copy
import time
from ctypes import ArgumentError

import torch_geometric.transforms as T
from torch_geometric.data import Data

import helpers
from setting import *


class SettingTorch(SettingDynamic):
    def __init__(self, mesh_size, mesh_type, corners, is_adaptive):
        super().__init__(mesh_size, mesh_type, corners, is_adaptive)

        self.AREA_torch = helpers.to_torch_double(self.AREA)
        self.B_torch = helpers.to_torch_double(self.B)

        self.A_plus_B_times_ts_torch = helpers.to_torch_double(self.A_plus_B_times_ts)
        self.C_cuda = helpers.to_torch_double(self.C).to(helpers.device)


    @property
    def contiguous_edges_torch(self):
        return helpers.to_torch_long(self.edges).t().contiguous()

    @property
    def initial_points_torch(self):
        return helpers.to_torch_float(self.initial_points)

    @property
    def normalized_initial_points_torch(self):
        return helpers.to_torch_float(self.normalized_initial_points)

    @property
    def normalized_points_torch(self):
        return helpers.to_torch_float(self.normalized_points)

    @property
    def edges_features_torch(self):
        return helpers.to_torch_float(self.edges_features)

    @property
    def normalized_forces_torch(self):
        return helpers.to_torch_double(self.normalized_forces)

    @property
    def normalized_u_old_torch(self):
        return helpers.to_torch_double(self.normalized_u_old)

    @property
    def normalized_v_old_torch(self):
        return helpers.to_torch_double(self.normalized_v_old)


    @property
    def input_u_old_torch(self):
        return helpers.to_torch_double(self.input_u_old)

    @property
    def input_v_old_torch(self):
        return helpers.to_torch_double(self.input_v_old)


    @property
    def normalized_E_cuda(self):
        return self.get_E(
            self.normalized_forces_torch,
            self.normalized_u_old_torch,
            self.normalized_v_old_torch,
            self.AREA_torch,
            self.A_plus_B_times_ts_torch,
            self.B_torch,
        ).to(helpers.device)

    @property
    def edges_data_torch(self):
        return helpers.to_torch_float(self.edges_data)

    def get_data_with_norm(self, data):
        return torch.hstack((data, torch.linalg.norm(data, keepdim=True, dim=1)))

    @property
    def angle_torch(self):
        return helpers.to_torch_float(self.angle)
        
    def L2_normalized_cuda(self, normalized_a_cuda):
        normalized_a_vector_cuda = helpers.stack_column(normalized_a_cuda)
        value = self.L2(
            normalized_a_vector_cuda.double(), self.C_cuda, self.normalized_E_cuda,
        )
        return value

    def L2_normalized_np(self, normalized_a):
        normalized_a_vector = helpers.stack_column(normalized_a)
        value = self.L2(
            normalized_a_vector, self.C, helpers.to_np(self.normalized_E_cuda)
        )
        return value

    def L2_normalized_nvt(self, normalized_a):  # np via torch
        normalized_a_cuda = helpers.to_torch_float(normalized_a).to(helpers.device)
        value_torch = self.L2_cuda(normalized_a_cuda)
        value = helpers.to_np(value_torch)
        return value.item()


    def rotate_torch(self, vectors, angle):
        s = torch.sin(angle)
        c = torch.cos(angle)

        rotated_vectors = torch.zeros_like(vectors)
        rotated_vectors[:, 0] = vectors[:, 0] * c - vectors[:, 1] * s
        rotated_vectors[:, 1] = vectors[:, 0] * s + vectors[:, 1] * c

        return rotated_vectors

    def rotate_to_upward_torch(self, vectors):
        return self.rotate_torch(vectors, self.angle_torch)

    def rotate_from_upward_torch(self, vectors):
        return self.rotate_torch(vectors, -self.angle_torch)
