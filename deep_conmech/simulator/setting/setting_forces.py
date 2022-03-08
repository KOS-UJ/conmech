import numba
import numpy as np
from conmech.helpers import nph
from deep_conmech.simulator.setting.setting_matrices import SettingMatrices
from numba import njit



def L2_new(a, C, E):
    a_vector = nph.stack_column(a)
    first = 0.5 * (C @ a_vector) - E
    value = first.reshape(-1) @ a_vector
    return value


@njit
def get_forces_by_function_numba(
    forces_function, initial_nodes, moved_nodes, scale_x, scale_y, current_time
):
    nodes_count = len(initial_nodes)
    forces = np.zeros((nodes_count, 2), dtype=numba.double)
    for i in range(nodes_count):
        initial_point = initial_nodes[i]
        moved_point = moved_nodes[i]
        forces[i] = forces_function(
            initial_point, moved_point, current_time, scale_x, scale_y
        )

    return forces


class SettingForces(SettingMatrices):
    def __init__(
        self,
        mesh_type,
        mesh_density_x,
        mesh_density_y=None,
        scale_x=None,
        scale_y=None,
        is_adaptive=False,
        create_in_subprocess=False,
    ):
        super().__init__(
            mesh_type,
            mesh_density_x,
            mesh_density_y,
            scale_x,
            scale_y,
            is_adaptive,
            create_in_subprocess,
        )
        self.forces = None

    def get_forces_by_function(self, forces_function, current_time):
        return get_forces_by_function_numba(
            numba.njit(forces_function),
            self.initial_nodes,
            self.moved_nodes,
            self.scale_x,
            self.scale_y,
            current_time,
        )

    @property
    def normalized_forces(self):
        return self.normalize_rotate(self.forces)

    def prepare(self, forces):
        super().prepare()
        self.forces = forces
        self.set_all_normalized_E_np()

    def clear(self):
        super().clear()
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
        t = self.boundary_nodes_count
        normalized_E_split = nph.unstack(self.normalized_E, self.dim)
        normalized_Et = nph.stack_column(normalized_E_split[:t, :])
        self.normalized_Ei = nph.stack_column(normalized_E_split[t:, :])
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
        F_vector = nph.stack_column(AREA @ forces)
        u_old_vector = nph.stack_column(u_old)
        v_old_vector = nph.stack_column(v_old)

        E = F_vector - A_plus_B_times_ts @ v_old_vector - B @ u_old_vector
        return E
