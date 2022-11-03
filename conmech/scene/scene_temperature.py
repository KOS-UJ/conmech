import jax.numpy as jnp
import numpy as np

from conmech.helpers import nph
from conmech.scene.body_forces import energy
from conmech.scene.scene import Scene
from conmech.solvers import SchurComplement


def obstacle_heat(
    penetration,
    tangential_velocity,
    heat_coeff,
):
    return (penetration > 0) * heat_coeff * nph.euclidean_norm(tangential_velocity, keepdims=True)


def integrate(
    nodes_normals,
    obstacle_normals,
    velocity,
    initial_penetration,
    nodes_volume,
    heat_coeff,
):
    _ = nodes_normals
    penetration = initial_penetration
    # get_penetration_norm(displacement_step, normals=nodes_normals, penetration)
    # v_tangential = nph.get_tangential(velocity, nodes_normals)

    v_tangential = nph.get_tangential(velocity, obstacle_normals)
    heat = obstacle_heat(penetration, v_tangential, heat_coeff)
    result = nodes_volume * heat
    return result


class SceneTemperature(Scene):
    def __init__(
        self,
        mesh_prop,
        body_prop,
        obstacle_prop,
        schedule,
        create_in_subprocess,
    ):
        super().__init__(
            mesh_prop=mesh_prop,
            body_prop=body_prop,
            obstacle_prop=obstacle_prop,
            schedule=schedule,
            create_in_subprocess=create_in_subprocess,
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

    def clear(self):
        super().clear()
        self.heat = None

    def set_temperature_old(self, temperature):
        self.t_old = temperature

    def iterate_self(self, acceleration, temperature=None):
        self.set_temperature_old(temperature)
        return super().iterate_self(acceleration=acceleration)

    def get_normalized_rhs_jax(self, temperature=None):
        value = super().get_normalized_rhs_jax()
        if temperature is not None:
            value += jnp.array(self.matrices.thermal_expansion.T @ temperature)
        return value

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
            # free_x_free_inverted=self.solver_cache.temperature_free_x_free_inv,
            free_x_free=self.solver_cache.temperature_free_x_free,
            contact_x_free=self.solver_cache.temperature_contact_x_free,
        )
        return normalized_t_rhs_boundary, normalized_t_rhs_free

    def get_normalized_t_rhs_jax(self, normalized_acceleration):
        U = self.matrices.acceleration_operator[self.independent_indices, self.independent_indices]

        v = jnp.array(self.normalized_velocity_old + normalized_acceleration * self.time_step)
        v_vector = nph.stack_column(v)

        A = nph.stack_column(self.matrices.volume_at_nodes @ self.heat)
        A += (-1) * self.matrices.thermal_expansion @ v_vector
        A += (1 / self.time_step) * U @ self.t_old

        obstacle_heat_integral = jnp.array(self.get_obstacle_heat_integral())
        A += self.complete_boundary_data_with_zeros(obstacle_heat_integral)
        return A

    # TODO: #65 Check why without new data !!!
    def get_obstacle_heat_integral(self):
        surface_per_boundary_node = self.get_surface_per_boundary_node()
        if self.has_no_obstacles:
            return np.zeros_like(surface_per_boundary_node)
        return integrate(
            nodes_normals=self.get_boundary_normals(),
            obstacle_normals=self.get_boundary_obstacle_normals(),
            velocity=self.boundary_velocity_old,
            initial_penetration=self.get_penetration_scalar(),
            nodes_volume=surface_per_boundary_node,
            heat_coeff=self.obstacle_prop.heat,
        )
