import numpy as np

from conmech.helpers import nph
from conmech.solvers import SchurComplement
from deep_conmech.simulator.setting import scene
from deep_conmech.simulator.setting.scene import Scene
from deep_conmech.simulator.setting.setting_forces import energy_new


def obstacle_heat(
    penetration_norm,
    tangential_velocity,
    heat_coeff,
):
    return (
        (penetration_norm > 0) * heat_coeff * nph.euclidean_norm(tangential_velocity, keepdims=True)
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
    penetration_norm = scene.get_penetration_norm(nodes, obstacle_nodes, obstacle_nodes_normals)
    v_tangential = nph.get_tangential(v, nodes_normals)

    heat = obstacle_heat(penetration_norm, v_tangential, heat_coeff)
    result = nodes_volume * heat
    return result


def energy_temperature(
    t,
    T,
    Q,
):
    return energy_new(t, T, Q)


class SceneTemperature(Scene):
    def __init__(
        self,
        mesh_prop,
        body_prop,
        obstacle_prop,
        schedule,
        normalize_by_rotation: bool,
        create_in_subprocess,
    ):
        super().__init__(
            mesh_prop=mesh_prop,
            body_prop=body_prop,
            obstacle_prop=obstacle_prop,
            schedule=schedule,
            normalize_by_rotation=normalize_by_rotation,
            create_in_subprocess=create_in_subprocess,
        )
        self.t_old = np.zeros((self.nodes_count, 1))
        self.heat = None

    def get_normalized_energy_temperature_np(self, normalized_a):
        normalized_Q_boundary, normalized_Q_free = self.get_all_normalized_Q_np(normalized_a)
        return (
            lambda normalized_boundary_t_vector: energy_temperature(
                nph.unstack(normalized_boundary_t_vector, 1),
                self.solver_cache.temperature_boundary,
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
            free_x_free_inverted=self.solver_cache.temperature_free_x_free_inv,
            contact_x_free=self.solver_cache.temperature_contact_x_free,
        )
        return normalized_Q_boundary, normalized_Q_free

    def get_normalized_Q_np(self, normalized_a):
        return self.get_Q(
            a=normalized_a,
            velocity_old=self.normalized_velocity_old,
            heat=self.heat,
            t_old=self.t_old,
            const_volume=self.volume,
            thermal_expansion=self.thermal_expansion,
            U=self.acceleration_operator[self.independent_indices, self.independent_indices],
            dimension=self.dimension,
            time_step=self.time_step,
        )

    def get_Q(
        self,
        a,
        velocity_old,
        heat,
        t_old,
        const_volume,
        thermal_expansion,
        U,
        dimension,
        time_step,
    ):
        v = velocity_old + a * time_step
        v_vector = nph.stack_column(v)

        Q = nph.stack_column(const_volume @ heat)
        Q += (-1) * thermal_expansion @ v_vector
        Q += (1 / self.time_step) * U @ t_old

        obstacle_heat_integral = self.get_obstacle_heat_integral()
        Q += self.complete_boundary_data_with_zeros(obstacle_heat_integral)
        return Q

    def get_obstacle_heat_integral(self):
        surface_per_boundary_node = self.get_surface_per_boundary_node()
        if self.has_no_obstacles:
            return np.zeros_like(surface_per_boundary_node)
        boundary_normals = self.get_boundary_normals()
        return integrate(
            nodes=self.boundary_nodes,
            nodes_normals=boundary_normals,
            obstacle_nodes=self.boundary_obstacle_nodes,
            obstacle_nodes_normals=self.boundary_obstacle_nodes_normals,
            v=self.boundary_velocity_old,
            nodes_volume=surface_per_boundary_node,
            heat_coeff=self.obstacle_prop.heat,
        )

    def get_normalized_E_np(self, t):
        return self.get_E(
            t=t,
            forces=self.normalized_forces,
            displacement_old=self.normalized_displacement_old,
            velocity_old=self.normalized_velocity_old,
            const_volume=self.volume,
            elasticity=self.elasticity,
            viscosity=self.viscosity,
            time_step=self.time_step,
            dimension=self.dimension,
            thermal_expansion=self.thermal_expansion,
        )

    def get_E(
        self,
        t,
        forces,
        displacement_old,
        velocity_old,
        const_volume,
        elasticity,
        viscosity,
        time_step,
        dimension,
        thermal_expansion,
    ):
        value = super().get_E(
            forces=forces,
            displacement_old=displacement_old,
            velocity_old=velocity_old,
            const_volume=const_volume,
            elasticity=elasticity,
            viscosity=viscosity,
            time_step=time_step,
        )
        value += thermal_expansion.T @ t
        return value

    def iterate_self(self, acceleration, t, randomized_inputs=False):
        self.set_t_old(t)
        return super().iterate_self(acceleration=acceleration)
