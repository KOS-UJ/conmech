import numpy as np

from conmech.helpers import nph
from conmech.scene import scene, setting_forces
from conmech.scene.scene import Scene
from conmech.scene.setting_forces import energy
from conmech.solvers import SchurComplement


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
    obstacle_normals,
    velocity,
    nodes_volume,
    heat_coeff,
):
    penetration_norm = scene.get_penetration_norm(nodes, obstacle_nodes, obstacle_normals)
    v_tangential = nph.get_tangential(velocity, nodes_normals)

    heat = obstacle_heat(penetration_norm, v_tangential, heat_coeff)
    result = nodes_volume * heat
    return result


def get_rhs(
    temperature,
    forces,
    displacement_old,
    velocity_old,
    const_volume,
    elasticity,
    viscosity,
    time_step,
    thermal_expansion,
):
    value = setting_forces.get_rhs(
        forces=forces,
        displacement_old=displacement_old,
        velocity_old=velocity_old,
        const_volume=const_volume,
        elasticity=elasticity,
        viscosity=viscosity,
        time_step=time_step,
    )
    value += thermal_expansion.T @ temperature
    return value


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
        normalized_Q_boundary, normalized_Q_free = self.get_all_normalized_t_rhs_np(normalized_a)
        return (
            lambda normalized_boundary_t_vector: energy(
                nph.unstack(normalized_boundary_t_vector, 1),
                self.solver_cache.temperature_boundary,
                normalized_Q_boundary,
            ),
            normalized_Q_free,
        )

    def prepare_tmp(self, forces, heat):
        self.prepare(forces)
        self.heat = heat

    def clear(self):
        super().clear()
        self.heat = None

    def set_temperature_old(self, temperature):
        self.t_old = temperature

    def get_all_normalized_t_rhs_np(self, normalized_a):
        normalized_t_rhs = self.get_normalized_t_rhs_np(normalized_a)
        (
            normalized_t_rhs_boundary,
            normalized_t_rhs_free,
        ) = SchurComplement.calculate_schur_complement_vector(
            vector=normalized_t_rhs,
            dimension=1,
            contact_indices=self.contact_indices,
            free_indices=self.free_indices,
            free_x_free_inverted=self.solver_cache.temperature_free_x_free_inv,
            contact_x_free=self.solver_cache.temperature_contact_x_free,
        )
        return normalized_t_rhs_boundary, normalized_t_rhs_free

    def get_normalized_t_rhs_np(self, normalized_a):
        return self.get_t_rhs(
            acceleration=normalized_a,
            velocity_old=self.normalized_velocity_old,
            heat=self.heat,
            t_old=self.t_old,
            const_volume=self.volume,
            thermal_expansion=self.thermal_expansion,
            U=self.acceleration_operator[self.independent_indices, self.independent_indices],
            time_step=self.time_step,
        )

    def get_t_rhs(
        self,
        acceleration,
        velocity_old,
        heat,
        t_old,
        const_volume,
        thermal_expansion,
        U,
        time_step,
    ):
        v = velocity_old + acceleration * time_step
        v_vector = nph.stack_column(v)

        A = nph.stack_column(const_volume @ heat)
        A += (-1) * thermal_expansion @ v_vector
        A += (1 / self.time_step) * U @ t_old

        obstacle_heat_integral = self.get_obstacle_heat_integral()
        A += self.complete_boundary_data_with_zeros(obstacle_heat_integral)
        return A

    def get_obstacle_heat_integral(self):
        surface_per_boundary_node = self.get_surface_per_boundary_node()
        if self.has_no_obstacles:
            return np.zeros_like(surface_per_boundary_node)
        boundary_normals = self.get_boundary_normals()
        boundary_obstacle_normals = self.get_boundary_obstacle_normals()
        return integrate(
            nodes=self.boundary_nodes,
            nodes_normals=boundary_normals,
            obstacle_nodes=self.boundary_obstacle_nodes,
            obstacle_normals=boundary_obstacle_normals,
            velocity=self.boundary_velocity_old,
            nodes_volume=surface_per_boundary_node,
            heat_coeff=self.obstacle_prop.heat,
        )

    def temp_get_normalized_rhs_np(self, temperature):
        return get_rhs(
            temperature=temperature,
            forces=self.normalized_forces,
            displacement_old=self.normalized_displacement_old,
            velocity_old=self.normalized_velocity_old,
            const_volume=self.volume,
            elasticity=self.elasticity,
            viscosity=self.viscosity,
            time_step=self.time_step,
            thermal_expansion=self.thermal_expansion,
        )

    def iterate_self_tmp(self, acceleration, temperature):
        self.set_temperature_old(temperature)
        return self.iterate_self(acceleration=acceleration)
