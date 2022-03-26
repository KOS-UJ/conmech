import numba
import numpy as np
from conmech.helpers import nph
from conmech.solvers.optimization.schur_complement import SchurComplement
from deep_conmech.simulator.setting.setting_matrices import SettingMatrices
from numba import njit


def L2_new(a, C, E):
    a_vector = nph.stack_column(a)
    first = 0.5 * (C @ a_vector) - E
    value = first.reshape(-1) @ a_vector
    return value


@njit
def get_forces_by_function_numba(
    forces_function, initial_nodes, moved_nodes, scale_x, scale_y, scale_z, current_time
):
    forces = np.zeros_like(initial_nodes, dtype=numba.double)
    for i in range(len(initial_nodes)):
        forces[i] = forces_function(
            initial_nodes[i], moved_nodes[i], current_time, scale_x, scale_y
        )
    return forces


class SettingForces(SettingMatrices):
    def __init__(
        self, mesh_data, body_prop, schedule, create_in_subprocess,
    ):
        super().__init__(
            mesh_data=mesh_data,
            body_prop=body_prop,
            schedule=schedule,
            create_in_subprocess=create_in_subprocess,
        )
        self.forces = None

    def get_forces_by_function(self, forces_function, current_time):
        return get_forces_by_function_numba(
            numba.njit(forces_function),
            self.initial_nodes,
            self.moved_nodes,
            self.mesh_data.scale_x,
            self.mesh_data.scale_y,
            self.mesh_data.scale_z,
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
        self.normalized_E_free = None
        self.normalized_E_boundary = None

    def set_all_normalized_E_np(self):
        self.normalized_E = self.get_normalized_E_np()
        (
            self.normalized_E_boundary,
            self.normalized_E_free,
        ) = SchurComplement.calculate_schur_complement_vector(
            vector=self.normalized_E,
            dimension=self.dimension,
            contact_indices=self.contact_indices,
            free_indices=self.free_indices,
            free_x_free_inverted=self.free_x_free_inverted,
            contact_x_free=self.contact_x_free,
        )

    def get_normalized_E_np(self):
        return self.get_E(
            self.normalized_forces,
            self.normalized_u_old,
            self.normalized_v_old,
            self.VOL,
            self.A_plus_B_times_ts,
            self.B,
        )

    def get_E(self, forces, u_old, v_old, VOL, A_plus_B_times_ts, B):
        F_vector = nph.stack_column(VOL @ forces)
        u_old_vector = nph.stack_column(u_old)
        v_old_vector = nph.stack_column(v_old)

        E = F_vector - A_plus_B_times_ts @ v_old_vector - B @ u_old_vector
        return E
