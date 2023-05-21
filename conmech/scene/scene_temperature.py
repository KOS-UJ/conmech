import jax
import jax.numpy as jnp
import numpy as np

from conmech.helpers import jxh, nph
from conmech.helpers.config import SimulationConfig
from conmech.scene.body_forces import energy
from conmech.scene.energy_functions import _get_penetration_positive
from conmech.scene.scene import Scene
from conmech.solvers import SchurComplement


def obstacle_heat(
    penetration,
    tangential_velocity,
    heat_coeff,
):
    return (penetration > 0) * heat_coeff * nph.euclidean_norm(tangential_velocity, keepdims=True)


def integrate_boundary_temperature(
    boundary_obstacle_normals,
    boundary_velocity_new,
    initial_penetration,
    nodes_volume,
    heat_coeff,
    time_step,
):
    boundary_displacement_step = time_step * boundary_velocity_new
    penetration_norm = _get_penetration_positive(
        displacement_step=boundary_displacement_step,
        normals=(-1) * boundary_obstacle_normals, # # TODO: Check this / boundary_obstacle_normals,
        initial_penetration=initial_penetration,
    )

    v_tangential = nph.get_tangential(boundary_velocity_new, boundary_obstacle_normals)  # nodes_normals
    heat = obstacle_heat(penetration_norm, v_tangential, heat_coeff)
    result = nodes_volume * heat
    return result


class SceneTemperature(Scene):
    def __init__(
        self,
        mesh_prop,
        body_prop,
        obstacle_prop,
        schedule,
        simulation_config: SimulationConfig,
        create_in_subprocess,
    ):
        super().__init__(
            mesh_prop=mesh_prop,
            body_prop=body_prop,
            obstacle_prop=obstacle_prop,
            schedule=schedule,
            create_in_subprocess=create_in_subprocess,
            simulation_config=simulation_config,
        )
        self.t_old = np.zeros((self.nodes_count, 1))
        self.heat = None

    def get_normalized_energy_temperature_np(self, normalized_acceleration):
        normalized_t_rhs_boundary, normalized_t_rhs_free = self.get_all_normalized_t_rhs_np(
            normalized_acceleration
        )
        return (
            lambda normalized_boundary_t_vector: energy(
                nph.unstack(normalized_boundary_t_vector, 1),
                self.solver_cache.temperature_boundary,
                normalized_t_rhs_boundary,
            ),
            normalized_t_rhs_free,
        )

    def prepare_tmp(self, forces, heat):
        self.prepare(forces)
        self.heat = heat

    def clear_external_factors(self):
        super().clear_external_factors()
        self.heat = None

    def set_temperature_old(self, temperature):
        self.t_old = temperature

    def iterate_self(self, acceleration, temperature=None):
        self.set_temperature_old(temperature)
        return super().iterate_self(acceleration=acceleration)

    def get_all_normalized_t_rhs_np(self, normalized_acceleration):
        normalized_t_rhs = self.get_normalized_t_rhs_jax(normalized_acceleration)
        (
            normalized_t_rhs_boundary,
            normalized_t_rhs_free,
        ) = SchurComplement.calculate_schur_complement_vector(
            vector=normalized_t_rhs,
            dimension=1,
            contact_indices=self.contact_indices,
            free_indices=self.free_indices,
            free_x_free_inverted=self.solver_cache.temperature_free_x_free_inv,
            # free_x_free=self.solver_cache.temperature_free_x_free,
            contact_x_free=self.solver_cache.temperature_contact_x_free,
        )
        return normalized_t_rhs_boundary, normalized_t_rhs_free

    def get_normalized_t_rhs_jax(self, normalized_acceleration):  # TODO: jax.jit
        U = self.matrices.acceleration_operator[self.independent_indices, self.independent_indices]

        velocity_new = jnp.array(
            self.normalized_velocity_old + normalized_acceleration * self.time_step
        )
        boundary_velocity_new = velocity_new[self.boundary_indices]
        velocity_new_vector = nph.stack_column(velocity_new)

        A = nph.stack_column(self.matrices.volume_at_nodes @ self.heat)
        A += (-1) * self.matrices.thermal_expansion @ velocity_new_vector
        A += (1 / self.time_step) * U @ self.t_old

        obstacle_heat_integral = jnp.array(self.get_obstacle_heat_integral(boundary_velocity_new))
        A += jxh.complete_data_with_zeros(data=obstacle_heat_integral, nodes_count=self.nodes_count)
        return A

    def get_obstacle_heat_integral(self, boundary_velocity_new):
        surface_per_boundary_node = self.get_surface_per_boundary_node_jax()
        if self.has_no_obstacles:
            return np.zeros_like(surface_per_boundary_node)
        return jax.jit(integrate_boundary_temperature)(
            boundary_obstacle_normals=self.boundary_obstacle_normals,
            boundary_velocity_new=boundary_velocity_new,
            initial_penetration=self.penetration_scalars,
            nodes_volume=surface_per_boundary_node,
            heat_coeff=self.obstacle_prop.heat,
            time_step=self.time_step,
        )
