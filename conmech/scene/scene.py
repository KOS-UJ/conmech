from typing import List, Optional

import jax
import jax.numpy as jnp
import numba
import numpy as np

from conmech.dynamics.dynamics import DynamicsConfiguration, SolverMatrices
from conmech.dynamics.factory.dynamics_factory_method import ConstMatrices
from conmech.helpers import jxh, lnh, nph
from conmech.helpers.config import SimulationConfig
from conmech.helpers.interpolation_helpers import get_interlayer_data_skinning
from conmech.helpers.lnh import get_in_base
from conmech.properties.body_properties import TimeDependentBodyProperties
from conmech.properties.mesh_properties import MeshProperties
from conmech.properties.obstacle_properties import ObstacleProperties
from conmech.properties.schedule import Schedule
from conmech.scene.body_forces import BodyForces
from conmech.scene.energy_functions import (
    EnergyObstacleArguments,
    _obstacle_resistance_potential_normal,
    _obstacle_resistance_potential_tangential,
)
from conmech.solvers.optimization.schur_complement import SchurComplement
from conmech.state.body_position import BodyPosition, mesh_normalization_decorator


@numba.njit
def get_closest_obstacle_to_boundary_numba(boundary_nodes, obstacle_nodes):
    boundary_obstacle_indices = np.zeros((len(boundary_nodes)), dtype=numba.int64)

    for i, boundary_node in enumerate(boundary_nodes):
        distances = nph.euclidean_norm_numba(obstacle_nodes - boundary_node)
        boundary_obstacle_indices[i] = distances.argmin()

    return boundary_obstacle_indices


# pylint: disable=R0904
class Scene(BodyForces):
    def __init__(
        self,
        mesh_prop: MeshProperties,
        body_prop: TimeDependentBodyProperties,
        obstacle_prop: ObstacleProperties,
        schedule: Schedule,
        simulation_config: SimulationConfig,
        with_schur: bool = False,
        create_in_subprocess: bool = False,
    ):
        super().__init__(
            mesh_prop=mesh_prop,
            body_prop=body_prop,
            schedule=schedule,
            simulation_config=simulation_config,
            dynamics_config=DynamicsConfiguration(
                create_in_subprocess=create_in_subprocess,
                with_lhs=False,
                with_schur=with_schur,
            ),
        )
        self.obstacle_prop = obstacle_prop
        self.closest_obstacle_indices = None
        self.linear_obstacles: np.ndarray = np.array([[], []])
        self.mesh_obstacles: List[BodyPosition] = []
        self.energy_functions = None
        self.lifted_acceleration = None
        self.clear()

    def prepare(self, inner_forces):
        super().prepare(inner_forces)
        if not self.has_no_obstacles:
            self.closest_obstacle_indices = get_closest_obstacle_to_boundary_numba(
                self.boundary_nodes, self.obstacle_nodes
            )

    def clean_acceleration(self, normalized_acceleration):
        _ = self
        if normalized_acceleration is None:
            return None
        return normalized_acceleration

    def normalize_and_set_obstacles(
        self,
        obstacles_unnormalized: Optional[np.ndarray],
        all_mesh_prop: Optional[List[MeshProperties]],
    ):
        if obstacles_unnormalized is not None and obstacles_unnormalized.size > 0:
            self.linear_obstacles = obstacles_unnormalized
            self.linear_obstacles[0, ...] = nph.normalize_euclidean_numba(
                self.linear_obstacles[0, ...]
            )
        if all_mesh_prop is not None:
            self.mesh_obstacles.extend(
                [BodyPosition(mesh_prop=mesh_prop, schedule=None) for mesh_prop in all_mesh_prop]
            )

    def get_energy_obstacle_args_for_jax(self, energy_functions, temperature=None):
        args, rhs_acceleration = self._get_initial_energy_obstacle_args_for_jax(temperature)
        args = EnergyObstacleArguments(
            lhs_acceleration_jax=self.solver_cache.lhs_acceleration_jax,  # .todense(),
            rhs_acceleration=rhs_acceleration,
            boundary_velocity_old=args.boundary_velocity_old,
            boundary_normals=args.boundary_normals,
            boundary_obstacle_normals=args.boundary_obstacle_normals,
            initial_penetration=args.initial_penetration,
            surface_per_boundary_node=args.surface_per_boundary_node,
            body_prop=args.body_prop,
            obstacle_prop=args.obstacle_prop,
            time_step=jnp.array(args.time_step),
            element_initial_volume=jnp.array(self.matrices.element_initial_volume),
            dx_big_jax=self.matrices.dx_big_jax,  # .todense(),
            base_displacement=args.base_displacement,
            base_energy_displacement=jax.jit(energy_functions.compute_displacement_energy)(
                displacement=args.base_displacement,
                dx_big_jax=self.matrices.dx_big_jax,
                element_initial_volume=self.matrices.element_initial_volume,
                body_prop=args.body_prop,
            ),
            base_velocity=args.base_velocity,
            base_energy_velocity=jax.jit(energy_functions.compute_velocity_energy)(
                velocity=args.base_velocity,
                dx_big_jax=self.matrices.dx_big_jax,
                element_initial_volume=self.matrices.element_initial_volume,
                body_prop=args.body_prop,
            ),
            displacement_old=args.displacement_old,
        )
        return args

    @property
    def linear_obstacle_nodes(self):
        return self.linear_obstacles[1, ...]

    @property
    def linear_obstacle_normals(self):
        return self.linear_obstacles[0, ...]

    @property
    def obstacle_nodes(self):
        all_nodes = []
        all_nodes.extend(list(self.linear_obstacle_nodes))
        all_nodes.extend([m.boundary_nodes for m in self.mesh_obstacles])
        return np.vstack(all_nodes)

    def get_obstacle_normals(self):
        all_normals = []
        all_normals.extend(list(self.linear_obstacle_normals))
        all_normals.extend([m.get_boundary_normals_jax() for m in self.mesh_obstacles])
        return np.vstack(all_normals)

    @property
    def boundary_obstacle_nodes(self):
        return self.obstacle_nodes[self.closest_obstacle_indices]

    @property
    def norm_boundary_obstacle_nodes(self):
        return self.normalize_rotate(self.boundary_obstacle_nodes - self.mean_moved_nodes)

    def get_norm_obstacle_normals(self):
        return self.normalize_rotate(self.get_obstacle_normals())

    def get_boundary_obstacle_normals_OBS(self):
        return self.get_obstacle_normals()[self.closest_obstacle_indices]

    def get_boundary_obstacle_normals(self):
        return -1.0 * self.get_boundary_normals_jax()

    def get_norm_boundary_obstacle_normals(self):
        return self.normalize_rotate(self.get_boundary_obstacle_normals())

    @property
    def normalized_obstacle_nodes(self):
        return self.normalize_rotate(self.obstacle_nodes - self.mean_moved_nodes)

    @property
    def boundary_velocity_old(self):
        return self.velocity_old[self.boundary_indices]

    @property
    def norm_boundary_velocity_old(self):
        return self.normalized_velocity_old[self.boundary_indices]

    @property
    def normalized_boundary_nodes(self):
        return self.normalized_nodes[self.boundary_indices]

    def get_penetration_scalar(self):
        collisions = (-1) * nph.elementwise_dot(
            (self.boundary_nodes - self.boundary_obstacle_nodes),
            self.get_boundary_obstacle_normals_OBS(),
        ).reshape(-1, 1)
        return self.apply_self_collisions(collisions)

    def apply_self_collisions(self, collisions):
        closest_nodes, closest_distances, closest_weights = get_interlayer_data_skinning(
            self.moved_nodes, self.elements, self.boundary_nodes
        )
        inside_mask = (closest_weights > 0.001).all(axis=1)
        if inside_mask.any():
            inside_nodes = nph.elementwise_dot(
                self.initial_nodes[closest_nodes[inside_mask]],
                closest_weights[inside_mask].reshape(-1, 4, 1),
            )
            boundary_indices = get_closest_obstacle_to_boundary_numba(
                inside_nodes, self.initial_boundary_nodes
            )
            collisions[inside_mask] = (-1) * nph.elementwise_dot(
                (self.boundary_nodes[boundary_indices] - inside_nodes),
                self.get_boundary_normals_jax()[boundary_indices],
            ).reshape(-1, 1)
        # collisions[inside_mask] = 0.1
        return collisions

    def get_penetration_positive(self):
        penetration = self.get_penetration_scalar()
        return penetration * (penetration > 0)

    def __get_boundary_penetration(self):
        return self.get_penetration_positive() * self.get_boundary_obstacle_normals()

    def get_normalized_boundary_penetration(self):
        return self.normalize_rotate(self.__get_boundary_penetration())

    def __get_boundary_v_tangential(self):
        return nph.get_tangential(self.boundary_velocity_old, self.get_boundary_normals_jax())

    def __get_normalized_boundary_v_tangential(self):
        return nph.get_tangential_numba(
            self.norm_boundary_velocity_old, np.array(self.get_normalized_boundary_normals_jax())
        )

    def __get_friction_vector(self):
        return (self.get_penetration_scalar() > 0) * np.nan_to_num(
            nph.normalize_euclidean_numba(self.__get_normalized_boundary_v_tangential())
        )

    def get_normal_response_input(self):
        return (
            self.obstacle_prop.hardness * self.get_penetration_positive()
        )  # self.get_normalized_boundary_penetration()

    def get_friction_input(self):
        return self.obstacle_prop.friction * self.__get_friction_vector()

    def get_resistance_normal(self):
        return _obstacle_resistance_potential_normal(
            self.get_penetration_positive(), self.obstacle_prop.hardness, self.time_step
        )

    def get_resistance_tangential(self):
        return _obstacle_resistance_potential_tangential(
            self.get_penetration_positive(),
            self.__get_boundary_v_tangential(),
            friction=self.obstacle_prop.friction,
            time_step=self.time_step,
            use_nonconvex_friction_law=self.simulation_config.use_nonconvex_friction_law,
        )

    @property
    def has_no_obstacles(self):
        return self.linear_obstacles.size == 0 and len(self.mesh_obstacles) == 0

    def _get_colliding_nodes_indicator(self):
        if self.has_no_obstacles:
            return np.zeros((self.nodes_count, 1), dtype=np.int64)
        return jxh.complete_data_with_zeros(
            data=(self.get_penetration_scalar() > 0) * 1, nodes_count=self.nodes_count
        )

    def is_colliding(self):
        return np.any(self._get_colliding_nodes_indicator())

    def prepare_to_save(self):
        self.energy_functions = None
        self.matrices = ConstMatrices()
        # lhs_sparse = self.solver_cache.lhs_sparse
        self.solver_cache = SolverMatrices()
        # self.solver_cache.lhs_sparse = lhs_sparse
        # self.reduced ...

    @property
    @mesh_normalization_decorator
    def normalized_exact_acceleration(self):
        return self.normalize_rotate(self.exact_acceleration)

    @property
    @mesh_normalization_decorator
    def normalized_lifted_acceleration(self):
        return self.normalize_rotate(self.lifted_acceleration)

    @mesh_normalization_decorator
    def force_denormalize(self, acceleration):
        return self.denormalize_rotate(acceleration)

    @property
    def norm_exact_new_displacement(self):
        return self.to_normalized_displacement(self.exact_acceleration)

    @property
    def norm_lifted_new_displacement(self):
        return self.to_normalized_displacement(self.lifted_acceleration)

    def to_displacement(self, acceleration):
        velocity_new = self.velocity_old + self.time_step * acceleration
        displacement_new = self.displacement_old + self.time_step * velocity_new
        return displacement_new

    def from_displacement(self, displacement):
        velocity = (displacement - self.displacement_old) / self.time_step
        acceleration = (velocity - self.velocity_old) / self.time_step
        return acceleration

    @mesh_normalization_decorator
    def to_normalized_displacement(self, acceleration):
        displacement_new = self.to_displacement(acceleration)

        moved_nodes_new = self.initial_nodes + displacement_new
        new_normalized_nodes = get_in_base(
            (moved_nodes_new - np.mean(moved_nodes_new, axis=0)),
            self.get_rotation(displacement_new),
        )
        return new_normalized_nodes - self.normalized_initial_nodes

    @mesh_normalization_decorator
    def to_normalized_displacement_rotated(self, acceleration):
        displacement_new = self.to_displacement(acceleration)

        moved_nodes_new = self.initial_nodes + displacement_new
        new_normalized_nodes = get_in_base(
            (moved_nodes_new - np.mean(moved_nodes_new, axis=0)),
            self.get_rotation(self.displacement_old),
        )
        assert np.allclose(new_normalized_nodes, self.normalize_shift_and_rotate(moved_nodes_new))
        return new_normalized_nodes - self.normalized_initial_nodes

    @mesh_normalization_decorator
    def to_normalized_displacement_rotated_displaced(self, acceleration):
        displacement_new = self.to_displacement(acceleration)

        moved_nodes_new = self.initial_nodes + displacement_new
        new_normalized_nodes = get_in_base(
            (moved_nodes_new - np.mean(self.moved_nodes, axis=0)),
            self.get_rotation(self.displacement_old),
        )
        assert np.allclose(
            new_normalized_nodes,
            self.normalize_rotate(moved_nodes_new - np.mean(self.moved_nodes, axis=0)),
        )
        return new_normalized_nodes - self.normalized_initial_nodes

    def get_last_displacement_step(self):
        return self.displacement_old - self.time_step * self.velocity_old

    def displacement_from_step(self, displacement_step):
        return self.displacement_old + displacement_step

    def get_centered_nodes(self, displacement):
        nodes = self.centered_initial_nodes + displacement
        centered_nodes = lnh.get_in_base(
            (nodes - nodes.mean(axis=0)), self.get_rotation(displacement)
        )
        return centered_nodes

    def get_displacement(self, base, position, base_displacement=None):
        if base_displacement is None:
            centered_nodes = self.centered_nodes
        else:
            centered_nodes = self.get_centered_nodes(base_displacement)
        moved_centered_nodes = lnh.get_in_base(centered_nodes, np.linalg.inv(base)) + position
        displacement = moved_centered_nodes - self.centered_initial_nodes
        return displacement

    def _get_initial_energy_obstacle_args_for_jax(self, temperature=None):
        base_velocity = self.normalized_velocity_old
        base_displacement = self.normalized_displacement_old + self.time_step * base_velocity

        args = EnergyObstacleArguments(
            lhs_acceleration_jax=None,
            rhs_acceleration=None,
            boundary_velocity_old=jnp.asarray(self.norm_boundary_velocity_old),
            boundary_normals=self.get_normalized_boundary_normals_jax(),
            boundary_obstacle_normals=jnp.asarray(self.get_norm_boundary_obstacle_normals()),
            initial_penetration=jnp.asarray(self.get_penetration_scalar()),
            surface_per_boundary_node=self.get_surface_per_boundary_node_jax(),
            body_prop=self.body_prop.get_tuple(),
            obstacle_prop=self.obstacle_prop,
            time_step=self.time_step,
            element_initial_volume=None,
            dx_big_jax=None,
            base_displacement=jnp.asarray(base_displacement),
            base_energy_displacement=None,
            base_velocity=jnp.asarray(base_velocity),
            base_energy_velocity=None,
            displacement_old=jnp.asarray(self.displacement_old),
        )
        rhs_acceleration = self.get_normalized_integrated_forces_column_for_jax(args)
        if temperature is not None:
            rhs_acceleration += jnp.array(self.matrices.thermal_expansion.T @ temperature)
        return args, rhs_acceleration

    def get_all_normalized_rhs_jax(self, temperature=None):
        normalized_rhs = self.get_normalized_rhs_jax(temperature)
        (
            normalized_rhs_boundary,
            normalized_rhs_free,
        ) = SchurComplement.calculate_schur_complement_vector(
            vector=normalized_rhs,
            dimension=self.dimension,
            contact_indices=self.contact_indices,
            free_indices=self.free_indices,
            free_x_free_inverted=self.solver_cache.free_x_free_inverted,
            # free_x_free=self.solver_cache.free_x_free,
            contact_x_free=self.solver_cache.contact_x_free,
        )
        return normalized_rhs_boundary, normalized_rhs_free

    def get_normalized_rhs_jax(self, temperature=None):
        displacement_old_vector = nph.stack_column(self.normalized_displacement_old)
        velocity_old_vector = nph.stack_column(self.normalized_velocity_old)
        _, f_vector = self._get_initial_energy_obstacle_args_for_jax(temperature=temperature)
        rhs = (
            f_vector
            - (self.matrices.viscosity + self.matrices.elasticity * self.time_step)
            @ jnp.array(velocity_old_vector)
            - self.matrices.elasticity @ jnp.array(displacement_old_vector)
        )
        return rhs
