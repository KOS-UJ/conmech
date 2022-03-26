import numpy as np
from conmech.helpers import nph
from deep_conmech.graph.setting.setting_randomized import SettingRandomized
from deep_conmech.scenarios import Scenario
from deep_conmech.simulator.setting.setting_forces import *


def L2_temperature(
    t, T, Q,
):
    value = L2_new(t, T, Q)
    return value


class SettingTemperature(SettingRandomized):
    def __init__(
        self, mesh_data, body_prop, obstacle_prop, schedule, create_in_subprocess,
    ):
        super().__init__(
            mesh_data=mesh_data,
            body_prop=body_prop,
            obstacle_prop=obstacle_prop,
            schedule=schedule,
            create_in_subprocess=create_in_subprocess,
        )
        self.t_old = np.zeros((self.nodes_count, 1))
        self.heat = None


    def get_normalized_L2_temperature_np(self):
        return lambda normalized_boundary_t_vector: L2_temperature(
            nph.unstack(normalized_boundary_t_vector, 1),
            self.T_boundary,
            self.normalized_Q_boundary,
        )

    def prepare(self, forces, heat):
        super().prepare(forces)
        self.heat = heat
        self.set_all_normalized_Q_np()

    def clear(self):
        super().clear()
        self.heat = None

    def set_t_old(self, t):
        self.t_old = t

    def set_a_old(self, a):
        self.clear_all_Q()
        super().set_a_old(a)

    def set_v_old(self, v):
        self.clear_all_Q()
        super().set_v_old(v)

    def set_u_old(self, u):
        self.clear_all_Q()
        super().set_u_old(u)

    def clear_all_Q(self):
        self.normalized_Q = None
        self.normalized_Q_free = None
        self.normalized_Q_boundary = None

    def set_all_normalized_Q_np(self):
        self.normalized_Q = self.get_normalized_Q_np()
        (
            self.normalized_Q_boundary,
            self.normalized_Q_free,
        ) = SchurComplement.calculate_schur_complement_vector(
            vector=self.normalized_Q,
            dimension=1,
            contact_indices=self.contact_indices,
            free_indices=self.free_indices,
            free_x_free_inverted=self.T_free_x_free_inverted,
            contact_x_free=self.T_contact_x_free,
        )

    def get_normalized_Q_np(self):
        return self.get_Q(
            heat=self.heat,
            t_old=self.t_old,
            v_old=self.normalized_v_old,
            VOL=self.VOL,
            C2T=self.C2T,
            U=self.ACC[self.independent_indices, self.independent_indices],
            dimension=self.dimension,
        )

    def get_Q(self, heat, t_old, v_old, VOL, C2T, U, dimension):
        v_old_vector = nph.stack_column(v_old)

        Q = nph.stack_column(VOL @ heat)
        Q += (-1) * nph.unstack_and_sum_columns(
            C2T @ v_old_vector, dim=dimension, keepdims=True
        )  # here v_old_vector is column vector
        Q += (1 / self.time_step) * U @ t_old
        return Q

    def get_normalized_E_np(self):
        return self.get_E(
            forces=self.normalized_forces,
            u_old=self.normalized_u_old,
            v_old=self.normalized_v_old,
            VOL=self.VOL,
            A_plus_B_times_ts=self.A_plus_B_times_ts,
            B=self.B,
            dimension=self.dimension,
            C2T=self.C2T,
        )

    def get_E(self, forces, u_old, v_old, VOL, A_plus_B_times_ts, B, dimension, C2T):
        value = super().get_E(forces, u_old, v_old, VOL, A_plus_B_times_ts, B)
        value += C2T.T @ np.tile(self.t_old, (dimension, 1))
        return value

    def iterate_self(self, a, t, randomized_inputs=False):
        self.set_t_old(t)
        return super().iterate_self(a=a, randomized_inputs=randomized_inputs)

    @staticmethod
    def get_setting(
        scenario: Scenario, randomize: bool = False, create_in_subprocess: bool = False
    ):
        setting = SettingTemperature(
            mesh_data=scenario.mesh_data,
            body_prop=scenario.body_prop,
            obstacle_prop=scenario.obstacle_prop,
            schedule=scenario.schedule,
            create_in_subprocess=create_in_subprocess,
        )
        setting.set_randomization(randomize)
        setting.set_obstacles(scenario.obstacles)
        return setting
