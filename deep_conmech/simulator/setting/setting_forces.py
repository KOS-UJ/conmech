import numpy as np
from numba import njit
from common import basic_helpers
from simulator.setting.setting_matrices import SettingFeatures


@njit
def L2_numba(argument, C, E):
    first = 0.5 * (C @ argument) - E
    value = first.reshape(-1) @ argument
    return value


@njit
def L2_full_np(a, C, E):
    a_vector = basic_helpers.stack_column_numba(a)
    value = L2_numba(a_vector, C, E)
    return value


class SettingForces(SettingFeatures):
    def __init__(
        self, mesh_density, mesh_type, scale, is_adaptive, create_in_subprocess
    ):
        super().__init__(
            mesh_density, mesh_type, scale, is_adaptive, create_in_subprocess
        )
        self.forces = None
        

    @property
    def normalized_forces(self):
        return self.rotate_to_upward(self.forces)



    def prepare(self, forces):
        self.forces = forces
        self.set_all_normalized_E_np()
        
    def clear(self):
        self.forces = None


    
    def set_a_old(self, a):
        self.clear_all_E()
        super().set_a_old(a)

    def set_v_old(self, v):
        self.clear_all_E()
        super().set_v_old(v)

    def set_u_old(self, u):
        self.clear_all_E()
        super().set_u_old(u)


    def clear_all_E(self):
        self.normalized_E = None
        self.normalized_Ei = None
        self.normalized_E_boundary = None

    def set_all_normalized_E_np(self):
        self.normalized_E = self.get_normalized_E_np()
        t = self.boundary_points_count
        normalized_E_split = basic_helpers.unstack(self.normalized_E)
        normalized_Et = basic_helpers.stack_column(normalized_E_split[:t, :])
        self.normalized_Ei = basic_helpers.stack_column(normalized_E_split[t:, :])
        CiiINVEi = self.CiiINV @ self.normalized_Ei
        self.normalized_E_boundary = normalized_Et - (self.Cti @ CiiINVEi)

    def get_normalized_E_np(self):
        return self.get_E(
            self.normalized_forces,
            self.normalized_u_old,
            self.normalized_v_old,
            self.AREA,
            self.A_plus_B_times_ts,
            self.B,
        )

    def get_E(self, forces, u_old, v_old, AREA, A_plus_B_times_ts, B):
        F_vector = basic_helpers.stack_column(AREA @ forces)
        u_old_vector = basic_helpers.stack_column(u_old)
        v_old_vector = basic_helpers.stack_column(v_old)

        E = F_vector - A_plus_B_times_ts @ v_old_vector - B @ u_old_vector
        return E

