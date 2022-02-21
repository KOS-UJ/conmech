import numpy as np
import scipy
import torch

import config
import helpers
from mesh_features import MeshFeatures


class Setting(MeshFeatures):

    def to_double_and_stack(self, x):
        return helpers.stack(x.double()).reshape(-1, 1)

    def L2(self, argument, C, E): #SAMPLE
        first = 0.5 * (C @ argument) - E
        value = first.reshape(-1) @ argument
        return value




class SettingDynamic(Setting):
    def __init__(self, mesh_size, mesh_type, corners, is_adaptive):
        super().__init__(mesh_size, mesh_type, corners, is_adaptive)
        self.set_forces(np.zeros_like(self.moved_points))



    @property
    def normalized_forces(self):
        return self.rotate_to_upward(self.forces)


    def set_forces(self, forces):
        self.forces = forces

    def set_forces_from_function(self, forces_function, time):
        forces = helpers.get_forces_by_function(forces_function, self, time)
        self.set_forces(forces)


    def get_E(self, forces, u_old, v_old, AREA, A_plus_B_times_ts, B):
        F_vector = helpers.stack_column(AREA @ forces)
        u_old_vector = helpers.stack_column(u_old)
        v_old_vector = helpers.stack_column(v_old)

        E = F_vector - A_plus_B_times_ts @ v_old_vector - B @ u_old_vector
        #E = F_vector + ACC_div_ts @ v_old_vector - B @ u_old_vector
        return E




    def get_E_np(self, forces, u_old, v_old):
        return self.get_E(forces, u_old, v_old, self.AREA, self.A_plus_B_times_ts, self.B)





class SettingStatic(Setting):
    def __init__(self, mesh_features):
        super().__init__(mesh_features)
        self.p = 1.0


    def L2_torch(self, u, forces):
        F = self.AREA_torch * forces.double()
        F_vector_torch = self.to_double_and_stack(F)
        u_vector = self.to_double_and_stack(u)

        C = self.B_torch.double()
        E = F_vector_torch.double()
        value = self.L2(u_vector, C, E)

        u_on_gamma = u_vector * self.on_gamma_d_stack_torch
        penality = self.p * (torch.sum(torch.abs(u_on_gamma)) ** 2)

        final_value = value + penality
        return final_value

    def L2_np(self, u, force):
        F = self.AREA * force

        F_vector = helpers.stack(F).reshape(-1, 1)
        u_vector = helpers.stack(u).reshape(-1, 1)

        C = self.B
        E = F_vector
        value = self.L2(u_vector, C, E)

        u_on_gamma = u_vector * self.on_gamma_d_stack
        penality = self.p * (np.sum(np.abs(u_on_gamma)) ** 2)

        final_value = value + penality
        return final_value.item()

    def calculate(self, forces):
        initial_vector = np.zeros([self.points_number, 2])
        return self.optimize(forces, initial_vector)
