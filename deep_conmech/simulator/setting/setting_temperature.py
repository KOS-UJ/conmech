import numpy as np
from conmech.helpers import nph
from conmech.helpers.config import Config
from deep_conmech.simulator.setting import setting_obstacles
from deep_conmech.simulator.setting.setting_forces import *
from deep_conmech.simulator.setting.setting_iterable import SettingIterable


def obstacle_heat(
    penetration_norm, tangential_velocity, heat_coeff,
):
    return (
        (penetration_norm > 0)
        * heat_coeff
        * nph.euclidean_norm(tangential_velocity, keepdims=True)
    )


def integrate(
    nodes,
    nodes_normals,
    obstacle_nodes,
    obstacle_nodes_normals,
    v,
    nodes_volume,
    heat_coeff,
):
    penetration_norm = setting_obstacles.get_penetration_norm(
        nodes, obstacle_nodes, obstacle_nodes_normals
    )
    v_tangential = nph.get_tangential(v, nodes_normals)

    heat = obstacle_heat(penetration_norm, v_tangential, heat_coeff)
    result = nodes_volume * heat
    return result


def L2_temperature(
    t, T, Q,
):
    return L2_new(t, T, Q)


class SettingTemperature(SettingIterable):
    def __init__(
        self,
        mesh_data,
        body_prop,
        obstacle_prop,
        schedule,
        normalize_by_rotation: bool,
        create_in_subprocess,
    ):
        super().__init__(
            mesh_data=mesh_data,
            body_prop=body_prop,
            obstacle_prop=obstacle_prop,
            schedule=schedule,
            normalize_by_rotation=normalize_by_rotation,
            create_in_subprocess=create_in_subprocess,
        )
        self.t_old = np.zeros((self.nodes_count, 1))
        self.heat = None

    def get_normalized_L2_temperature_np(self, normalized_a):
        normalized_Q_boundary, normalized_Q_free = self.get_all_normalized_Q_np(
            normalized_a
        )
        return (
            lambda normalized_boundary_t_vector: L2_temperature(
                nph.unstack(normalized_boundary_t_vector, 1),
                self.T_boundary,
                normalized_Q_boundary,
            ),
            normalized_Q_free,
        )

    def prepare(self, forces, heat):
        super().prepare(forces)
        self.heat = heat

    def clear(self):
        super().clear()
        self.heat = None

    def set_t_old(self, t):
        self.t_old = t

    def get_all_normalized_Q_np(self, normalized_a):
        normalized_Q = self.get_normalized_Q_np(normalized_a)
        (
            normalized_Q_boundary,
            normalized_Q_free,
        ) = SchurComplement.calculate_schur_complement_vector(
            vector=normalized_Q,
            dimension=1,
            contact_indices=self.contact_indices,
            free_indices=self.free_indices,
            free_x_free_inverted=self.T_free_x_free_inverted,
            contact_x_free=self.T_contact_x_free,
        )
        return normalized_Q_boundary, normalized_Q_free

    def get_normalized_Q_np(self, normalized_a):
        return self.get_Q(
            a=normalized_a,
            v_old=self.normalized_v_old,
            heat=self.heat,
            t_old=self.t_old,
            const_volume=self.const_volume,
            C2T=self.C2T,
            U=self.ACC[self.independent_indices, self.independent_indices],
            dimension=self.dimension,
            time_step=self.time_step,
        )

    def get_Q(self, a, v_old, heat, t_old, const_volume, C2T, U, dimension, time_step):
        v = v_old + a * time_step
        v_vector = nph.stack_column(v)

        Q = nph.stack_column(const_volume @ heat)
        Q += (-1) * nph.unstack_and_sum_columns(
            C2T @ v_vector, dim=dimension, keepdims=True
        )  # here v_old_vector is column vector
        Q += (1 / self.time_step) * U @ t_old

        obstacle_heat_integral = self.get_obstacle_heat_integral()
        Q += self.complete_boundary_data_with_zeros(obstacle_heat_integral)
        return Q

    def get_obstacle_heat_integral(self):
        return integrate(
            nodes=self.boundary_nodes,
            nodes_normals=self.boundary_normals,
            obstacle_nodes=self.boundary_obstacle_nodes,
            obstacle_nodes_normals=self.boundary_obstacle_normals,
            v=self.boundary_v_old,
            nodes_volume=self.boundary_nodes_volume,
            heat_coeff=0.01,  ##########################################
        )

    def get_normalized_E_np(self, t):
        return self.get_E(
            t=t,
            forces=self.normalized_forces,
            u_old=self.normalized_u_old,
            v_old=self.normalized_v_old,
            const_volume=self.const_volume,
            const_elasticity=self.const_elasticity,
            const_viscosity=self.const_viscosity,
            time_step=self.time_step,
            dimension=self.dimension,
            C2T=self.C2T,
        )

    def get_E(
        self,
        t,
        forces,
        u_old,
        v_old,
        const_volume,
        const_elasticity,
        const_viscosity,
        time_step,
        dimension,
        C2T,
    ):
        value = super().get_E(
            forces=forces,
            u_old=u_old,
            v_old=v_old,
            const_volume=const_volume,
            const_elasticity=const_elasticity,
            const_viscosity=const_viscosity,
            time_step=time_step,
        )
        value += C2T.T @ np.tile(t, (dimension, 1))
        return value

    def iterate_self(self, a, t, randomized_inputs=False):
        self.set_t_old(t)
        return super().iterate_self(a=a)
