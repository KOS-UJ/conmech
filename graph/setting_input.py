import copy
import time
from ctypes import ArgumentError

import torch_geometric.transforms as T
from torch_geometric.data import Data
from torch_geometric.utils import remove_self_loops
from calculator import Calculator
from setting import *
from setting_torch import SettingTorch


class SettingInput(SettingTorch):
    def __init__(self, mesh_size, mesh_type, corners, is_adaptive, randomized_inputs):
        super().__init__(mesh_size, mesh_type, corners, is_adaptive)
        self.set_randomization(randomized_inputs)


    def set_randomization(self, randomized_inputs):
        self.randomized_inputs = randomized_inputs
        if(randomized_inputs):
            self.v_old_randomization = helpers.get_random_normal(
                self.points_number, config.V_IN_RANDOM_FACTOR
            )
            self.u_old_randomization = helpers.get_random_normal(
                self.points_number, config.U_IN_RANDOM_FACTOR
            )
        else:            
            self.v_old_randomization = np.zeros_like(self.initial_points)
            self.u_old_randomization = np.zeros_like(self.initial_points)


    @property
    def normalized_v_old_randomization(self):
        return self.rotate_to_upward(self.v_old_randomization)

    @property
    def normalized_u_old_randomization(self):
        return self.rotate_to_upward(self.u_old_randomization)

    @property
    def randomized_v_old(self):
        return self.v_old + self.v_old_randomization

    @property
    def randomized_u_old(self):
        return self.u_old + self.u_old_randomization

    @property
    def input_v_old(self):  # normalized_randomized_v_old
        return self.normalized_v_old + self.normalized_v_old_randomization

    @property
    def input_u_old(self):  # normalized_randomized_u_old
        return self.normalized_u_old + self.normalized_u_old_randomization

    @property
    def input_u_old_torch(self):
        return helpers.to_torch_double(self.input_u_old)

    @property
    def input_v_old_torch(self):
        return helpers.to_torch_double(self.input_v_old)



    @property
    def a_correction(self):
        u_correction = config.U_NOISE_GAMMA * (self.u_old_randomization / (config.TIMESTEP * config.TIMESTEP))
        v_correction = (1. - config.U_NOISE_GAMMA) * self.v_old_randomization / config.TIMESTEP
        return -1.0 * (u_correction + v_correction)

    @property
    def normalized_a_correction(self):
        return self.rotate_to_upward(self.a_correction)

    @property
    def normalized_a_correction_cuda(self):
        return helpers.to_torch_float(self.normalized_a_correction).to(helpers.device)




    def make_dirty(self):
        self.v_old = self.randomized_v_old
        self.u_old = self.randomized_u_old

        self.v_old_randomization = np.zeros_like(self.initial_points)
        self.u_old_randomization = np.zeros_like(self.initial_points)
        self.randomized_inputs = False



    def get_copy(self):
        setting = copy.deepcopy(self)
        return setting
        

    def iterate_self(self, a, randomized_inputs):
        v = self.v_old + config.TIMESTEP * a
        u = self.u_old + config.TIMESTEP * v

        self.set_randomization(randomized_inputs)
        self.set_u_old(u)
        self.set_v_old(v)
        self.set_a_old(a)
        return self


    # MIN AT
    # a = a_cleaned - ((v_old - randomized_v_old) / config.TIMESTEP
    def L2_normalized_cuda(self, cleaned_normalized_a_cuda):
        return super().L2_normalized_cuda(cleaned_normalized_a_cuda - self.normalized_a_correction_cuda)

    def L2_normalized_np(self, cleaned_normalized_a):
        return super().L2_normalized_np(cleaned_normalized_a - self.normalized_a_correction)
       
    def L2_normalized_nvt(self, normalized_a):  # np via torch
        normalized_a_cuda = helpers.to_torch_float(normalized_a).to(helpers.device)
        value_torch = super().L2_normalized_cuda(normalized_a_cuda)
        value = helpers.to_np(value_torch)
        return value.item()

    @property
    def stacked_x(self):
        data = torch.hstack(
            (
                self.get_data_with_norm(self.normalized_forces_torch),
                self.get_data_with_norm(self.input_u_old_torch),
                self.get_data_with_norm(self.input_v_old_torch),
            )
        )
        return data.float()

    def get_data(self, setting_id=None, normalized_a_torch=None):
        edge_index, edge_attr = remove_self_loops(self.contiguous_edges_torch, self.edges_data_torch)
        data = Data(
            pos=self.normalized_initial_points_torch,
            x=self.stacked_x,
            edge_index=edge_index,
            edge_attr=edge_attr,  ##setting.edges_features_torch,
            setting=self,
            setting_id=setting_id,
            normalized_a_torch = normalized_a_torch
            # pin_memory=True,
            # num_workers=1
        )
        transform = T.Compose(
            [T.TargetIndegree(), T.Cartesian (), T.Polar()] # add custom for multiple 'pos' types
        )  # T.OneHotDegree(),
        transform(data)
        return data
