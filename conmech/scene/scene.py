from dataclasses import dataclass
from typing import List, NamedTuple, Optional

import jax
import jax.numpy as jnp
import numba
import numpy as np

from conmech.dynamics.dynamics import (
    DynamicsConfiguration,
    SolverMatrices,
    _get_deform_grad,
    _get_rotation_jax,
)
from conmech.dynamics.factory.dynamics_factory_method import ConstMatrices
from conmech.helpers import lnh, nph
from conmech.helpers.lnh import complete_base, get_in_base
from conmech.mesh.mesh import Mesh
from conmech.properties.body_properties import DynamicBodyProperties
from conmech.properties.mesh_properties import MeshProperties
from conmech.properties.obstacle_properties import ObstacleProperties
from conmech.properties.schedule import Schedule
from conmech.scene.body_forces import BodyForces
from conmech.state.body_position import BodyPosition
from deep_conmech.training_config import NORMALIZE, USE_GREEN_STRAIN


@numba.njit
def get_closest_obstacle_to_boundary_numba(boundary_nodes, obstacle_nodes):
    boundary_obstacle_indices = np.zeros((len(boundary_nodes)), dtype=numba.int64)

    for i, boundary_node in enumerate(boundary_nodes):
        distances = nph.euclidean_norm_numba(obstacle_nodes - boundary_node)
        boundary_obstacle_indices[i] = distances.argmin()

    return boundary_obstacle_indices


def _get_penetration_positive(displacement_step, normals, penetration):
    projection = nph.elementwise_dot(displacement_step, normals, keepdims=True) + penetration
    return (projection > 0) * projection


def _obstacle_resistance_potential_normal(penetration_norm, hardness, time_step):
    return hardness * 0.5 * (penetration_norm**2) * ((1.0 / time_step) ** 2)


def _obstacle_resistance_potential_tangential(
    penetration_norm,
    tangential_velocity,
    friction,
    time_step,
):
    return (
        (penetration_norm > 0)
        * friction
        * nph.euclidean_norm(tangential_velocity, keepdims=True)
        * (1.0 / time_step)
    )


class EnergyObstacleArguments(NamedTuple):
    lhs_acceleration_jax: np.ndarray
    rhs_acceleration: np.ndarray
    boundary_velocity_old: np.ndarray
    boundary_normals: np.ndarray
    boundary_obstacle_normals: np.ndarray
    penetration: np.ndarray
    surface_per_boundary_node: np.ndarray
    body_prop: np.ndarray
    obstacle_prop: np.ndarray
    time_step: float
    element_initial_volume: np.ndarray
    dx_big_jax: np.ndarray
    base_displacement: np.ndarray
    base_velocity: np.ndarray
    base_energy_displacement: np.ndarray
    base_energy_velocity: np.ndarray


def _get_boundary_integral(
    acceleration,
    args: EnergyObstacleArguments,
):
    boundary_nodes_count = args.boundary_velocity_old.shape[0]
    boundary_a = acceleration[:boundary_nodes_count, :]  # TODO: boundary slice

    boundary_v_new = args.boundary_velocity_old + args.time_step * boundary_a
    boundary_displacement_step = args.time_step * boundary_v_new

    normals = args.boundary_normals
    nodes_volume = args.surface_per_boundary_node
    hardness = args.obstacle_prop.hardness
    friction = args.obstacle_prop.friction

    penetration_norm = _get_penetration_positive(
        displacement_step=boundary_displacement_step,
        normals=normals,
        penetration=args.penetration,
    )
    velocity_tangential = nph.get_tangential(boundary_v_new, normals)

    resistance_normal = _obstacle_resistance_potential_normal(
        penetration_norm=penetration_norm, hardness=hardness, time_step=args.time_step
    )
    resistance_tangential = _obstacle_resistance_potential_tangential(
        penetration_norm=args.penetration,
        tangential_velocity=velocity_tangential,
        friction=friction,
        time_step=args.time_step,
    )
    boundary_integral = (nodes_volume * (resistance_normal + resistance_tangential)).sum()
    return boundary_integral


def _get_strain_lin(deform_grad):
    dimension = deform_grad.shape[1]
    identity = jnp.eye(dimension)
    deform_grad_t = deform_grad.transpose((0, 2, 1))
    return 0.5 * (deform_grad + deform_grad_t) - identity


def _get_strain_green(deform_grad):
    dimension = deform_grad.shape[1]
    identity = jnp.eye(dimension)
    deform_grad_t = deform_grad.transpose((0, 2, 1))
    return 0.5 * (deform_grad_t @ deform_grad - identity)


def _compute_component_energy(
    component,
    dx_big_jax,
    element_initial_volume,
    prop_1,
    prop_2,
    use_green_strain,
):
    f_w = _get_deform_grad(component, dx_big_jax)
    if use_green_strain:
        eps_w = _get_strain_green(deform_grad=f_w)
    else:
        eps_w = _get_strain_lin(deform_grad=f_w)

    phi = prop_1 * (eps_w * eps_w).sum(axis=(1, 2)) + (prop_2 / 2.0) * (
        ((eps_w).trace(axis1=1, axis2=2) ** 2)
    )
    energy = element_initial_volume @ phi
    return energy


def _compute_displacement_energy(
    displacement, dx_big_jax, element_initial_volume, body_prop, use_green_strain
):
    return _compute_component_energy(
        component=displacement,
        dx_big_jax=dx_big_jax,
        element_initial_volume=element_initial_volume,
        prop_1=body_prop.mu,
        prop_2=body_prop.lambda_,
        use_green_strain=use_green_strain,
    )


def _compute_velocity_energy(
    velocity, dx_big_jax, element_initial_volume, body_prop, use_green_strain
):
    return _compute_component_energy(
        component=velocity,
        dx_big_jax=dx_big_jax,
        element_initial_volume=element_initial_volume,
        prop_1=body_prop.theta,
        prop_2=body_prop.zeta,
        use_green_strain=use_green_strain,
    )


def _compute_energy(acceleration, args, use_green_strain):
    new_displacement = args.base_displacement + acceleration * args.time_step**2
    new_velocity = args.base_velocity + acceleration * args.time_step

    energy_new = (
        _compute_displacement_energy(
            displacement=new_displacement,
            dx_big_jax=args.dx_big_jax,
            element_initial_volume=args.element_initial_volume,
            body_prop=args.body_prop,
            use_green_strain=use_green_strain,
        )
        - args.base_energy_displacement
    ) / (args.time_step**2)

    energy_new += (
        _compute_velocity_energy(
            velocity=new_velocity,
            dx_big_jax=args.dx_big_jax,
            element_initial_volume=args.element_initial_volume,
            body_prop=args.body_prop,
            use_green_strain=use_green_strain,
        )
        - args.base_energy_velocity
    ) / (args.time_step)

    return energy_new


def _energy_vector(value_vector, lhs, rhs):
    first = 0.5 * (lhs @ value_vector) - rhs
    value = first.reshape(-1) @ value_vector
    return value[0]


def _energy_obstacle(acceleration_vector, args: EnergyObstacleArguments, use_green_strain: bool):
    # TODO: Repeat if collision
    dimension = args.base_displacement.shape[1]
    main_energy0 = _energy_vector(
        value_vector=nph.stack_column(acceleration_vector),
        lhs=args.lhs_acceleration_jax,
        rhs=args.rhs_acceleration,
    )
    main_energy1 = _compute_energy(
        acceleration=nph.unstack(acceleration_vector, dim=dimension),
        args=args,
        use_green_strain=use_green_strain,
    )
    return main_energy0 + main_energy1


def _energy_obstacle_colliding(
    acceleration_vector, args: EnergyObstacleArguments, use_green_strain: bool
):
    # TODO: Repeat if collision
    dimension = args.base_displacement.shape[1]
    main_energy = _energy_obstacle(
        acceleration_vector=acceleration_vector, args=args, use_green_strain=use_green_strain
    )
    acceleration = nph.unstack(acceleration_vector, dim=dimension)
    boundary_integral = _get_boundary_integral(acceleration=acceleration, args=args)
    return main_energy + boundary_integral


# @partial(jax.jit, static_argnums=(2,))
# energy_obstacle_jax = jax.jit(energy_obstacle)

# hes_energy_obstacle_new = jax.jit(
#     lambda x, args: (lambda f, x, v: jax.grad(lambda x: jnp.vdot(jax.grad(f)(x, args), v))(x))(
#         energy_obstacle, x, x
#     )
# )

# hes_energy_obstacle_colliding_new = jax.jit(
#     lambda x, args: (lambda f, x, v: jax.grad(lambda x: jnp.vdot(jax.grad(f)(x, args), v))(x))(
#         energy_obstacle_colliding_jax, x, x
#     )
# )


@dataclass
class EnergyFunctions:
    def __init__(self, use_green_strain):
        self.energy_obstacle_jax = jax.jit(
            lambda acceleration_vector, args: _energy_obstacle(
                acceleration_vector=acceleration_vector,
                args=args,
                use_green_strain=use_green_strain,
            )
        )
        self.energy_obstacle_colliding_jax = jax.jit(
            lambda acceleration_vector, args: _energy_obstacle_colliding(
                acceleration_vector=acceleration_vector,
                args=args,
                use_green_strain=use_green_strain,
            )
        )
        self.compute_displacement_energy_jax = jax.jit(
            lambda displacement, dx_big_jax, element_initial_volume, body_prop: _compute_displacement_energy(
                displacement=displacement,
                dx_big_jax=dx_big_jax,
                element_initial_volume=element_initial_volume,
                body_prop=body_prop,
                use_green_strain=use_green_strain,
            )
        )
        self.compute_velocity_energy_jax = jax.jit(
            lambda velocity, dx_big_jax, element_initial_volume, body_prop: _compute_velocity_energy(
                velocity=velocity,
                dx_big_jax=dx_big_jax,
                element_initial_volume=element_initial_volume,
                body_prop=body_prop,
                use_green_strain=use_green_strain,
            )
        )


class Scene(BodyForces):
    def __init__(
        self,
        mesh_prop: MeshProperties,
        body_prop: DynamicBodyProperties,
        obstacle_prop: ObstacleProperties,
        schedule: Schedule,
        create_in_subprocess: bool,
        with_schur: bool = False,
    ):
        super().__init__(
            mesh_prop=mesh_prop,
            body_prop=body_prop,
            schedule=schedule,
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
        self._energy_functions = None
        self.clear()

    def prepare(self, inner_forces):
        super().prepare(inner_forces)
        if not self.has_no_obstacles:
            self.closest_obstacle_indices = get_closest_obstacle_to_boundary_numba(
                self.boundary_nodes, self.obstacle_nodes
            )

    def get_energy_functions(self):
        if not self._energy_functions:
            self._energy_functions = EnergyFunctions(self.use_green_strain)
        return self._energy_functions

    def get_energy_function(self):
        if not self.is_colliding():
            return self.get_energy_functions().energy_obstacle_jax
        else:
            return self.get_energy_functions().energy_obstacle_colliding_jax

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

    def get_energy_obstacle_args_for_jax(self, temperature=None):
        _ = temperature

        displacement = self.normalized_displacement_old
        velocity = self.normalized_velocity_old

        base_displacement = displacement + self.time_step * velocity
        body_prop = self.body_prop.get_tuple()

        rhs_acceleration = self.get_normalized_integrated_forces_column_for_jax()
        if temperature is not None:
            rhs_acceleration += jnp.array(self.matrices.thermal_expansion.T @ temperature)

        # self.use_green_strain
        args = EnergyObstacleArguments(
            lhs_acceleration_jax=self.solver_cache.lhs_acceleration_jax,
            rhs_acceleration=rhs_acceleration,
            boundary_velocity_old=jnp.asarray(self.norm_boundary_velocity_old),
            boundary_normals=jnp.asarray(self.get_normalized_boundary_normals()),
            boundary_obstacle_normals=jnp.asarray(self.get_norm_boundary_obstacle_normals()),
            penetration=jnp.asarray(self.get_penetration_scalar()),
            surface_per_boundary_node=jnp.asarray(self.get_surface_per_boundary_node()),
            body_prop=body_prop,
            obstacle_prop=self.obstacle_prop,
            time_step=self.time_step,
            base_displacement=base_displacement,
            element_initial_volume=self.matrices.element_initial_volume,
            dx_big_jax=self.matrices.dx_big_jax,
            base_energy_displacement=self.get_energy_functions().compute_displacement_energy_jax(
                displacement=base_displacement,
                dx_big_jax=self.matrices.dx_big_jax,
                element_initial_volume=self.matrices.element_initial_volume,
                body_prop=body_prop,
            ),
            base_velocity=velocity,
            base_energy_velocity=self.get_energy_functions().compute_velocity_energy_jax(
                velocity=velocity,
                dx_big_jax=self.matrices.dx_big_jax,
                element_initial_volume=self.matrices.element_initial_volume,
                body_prop=body_prop,
            ),
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
        all_normals.extend([m.get_boundary_normals() for m in self.mesh_obstacles])
        return np.vstack(all_normals)

    @property
    def boundary_obstacle_nodes(self):
        return self.obstacle_nodes[self.closest_obstacle_indices]

    @property
    def norm_boundary_obstacle_nodes(self):
        return self.normalize_rotate(self.boundary_obstacle_nodes - self.mean_moved_nodes)

    def get_norm_obstacle_normals(self):
        return self.normalize_rotate(self.get_obstacle_normals())

    def get_boundary_obstacle_normals(self):
        return self.get_obstacle_normals()[self.closest_obstacle_indices]

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
        return (-1) * nph.elementwise_dot(
            (self.normalized_boundary_nodes - self.norm_boundary_obstacle_nodes),
            self.get_norm_boundary_obstacle_normals(),
        ).reshape(-1, 1)

    def get_penetration_positive(self):
        penetration = self.get_penetration_scalar()
        return penetration * (penetration > 0)

    def __get_boundary_penetration(self):
        return self.get_penetration_positive() * self.get_boundary_obstacle_normals()

    def get_normalized_boundary_penetration(self):
        return self.normalize_rotate(self.__get_boundary_penetration())

    def __get_boundary_v_tangential(self):
        return nph.get_tangential(self.boundary_velocity_old, self.get_boundary_normals())

    def __get_normalized_boundary_v_tangential(self):
        return nph.get_tangential_numba(
            self.norm_boundary_velocity_old, self.get_normalized_boundary_normals()
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
            self.obstacle_prop.friction,
            self.time_step,
        )

    @staticmethod
    def complete_mesh_boundary_data_with_zeros(mesh: Mesh, data: np.ndarray):
        return np.pad(data, ((0, mesh.nodes_count - len(data)), (0, 0)), "constant")

    def complete_boundary_data_with_zeros(self, data: np.ndarray):
        return Scene.complete_mesh_boundary_data_with_zeros(self, data)

    @property
    def has_no_obstacles(self):
        return self.linear_obstacles.size == 0 and len(self.mesh_obstacles) == 0

    def _get_colliding_nodes_indicator(self):
        if self.has_no_obstacles:
            return np.zeros((self.nodes_count, 1), dtype=np.int64)
        return self.complete_boundary_data_with_zeros((self.get_penetration_scalar() > 0) * 1)

    def is_colliding(self):
        return np.any(self._get_colliding_nodes_indicator())

    def prepare_to_save(self):
        self._energy_functions = None
        self.matrices = ConstMatrices()
        # lhs_sparse = self.solver_cache.lhs_sparse
        self.solver_cache = SolverMatrices()
        # self.solver_cache.lhs_sparse = lhs_sparse
        # self.reduced ...

    @property
    @Mesh.normalization_decorator
    def normalized_exact_acceleration(self):
        return self.normalize_rotate(self.exact_acceleration)

    @property
    @Mesh.normalization_decorator
    def normalized_lifted_acceleration(self):
        return self.normalize_rotate(self.lifted_acceleration)

    @Mesh.normalization_decorator
    def force_denormalize(self, acceleration):
        return self.denormalize_rotate(acceleration)

    @property
    def new_normalized_displacement(self):
        return self.to_normalized_displacement(self.exact_acceleration)

    @property
    def new_normalized_lifted_displacement(self):
        return self.to_normalized_displacement(self.lifted_acceleration)

    def to_displacement(self, acceleration):
        velocity_new = self.velocity_old + self.time_step * acceleration
        displacement_new = self.displacement_old + self.time_step * velocity_new
        return displacement_new

    @Mesh.normalization_decorator
    def to_normalized_displacement(self, acceleration):
        velocity_new = self.velocity_old + self.time_step * acceleration
        displacement_new = self.displacement_old + self.time_step * velocity_new

        moved_nodes_new = self.initial_nodes + displacement_new
        new_normalized_nodes = get_in_base(
            (moved_nodes_new - np.mean(moved_nodes_new, axis=0)),
            self.get_rotation(displacement_new),
        )
        return new_normalized_nodes - self.normalized_initial_nodes

    @Mesh.normalization_decorator
    def to_normalized_displacement_rotated(self, acceleration):
        velocity_new = self.velocity_old + self.time_step * acceleration
        displacement_new = self.displacement_old + self.time_step * velocity_new

        moved_nodes_new = self.initial_nodes + displacement_new
        new_normalized_nodes = get_in_base(
            (moved_nodes_new - np.mean(moved_nodes_new, axis=0)),
            self.get_rotation(self.displacement_old),
        )
        assert np.allclose(new_normalized_nodes, self.normalize_shift_and_rotate(moved_nodes_new))
        return new_normalized_nodes - self.normalized_initial_nodes

    @Mesh.normalization_decorator
    def to_normalized_displacement_rotated_displaced(self, acceleration):
        velocity_new = self.velocity_old + self.time_step * acceleration
        displacement_new = self.displacement_old + self.time_step * velocity_new

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

    def from_displacement(self, displacement):
        velocity = (displacement - self.displacement_old) / self.time_step
        acceleration = (velocity - self.velocity_old) / self.time_step
        return acceleration

    def get_centered_nodes(self, displacement):
        nodes = self.centered_initial_nodes + displacement
        centered_nodes = lnh.get_in_base(
            (nodes - np.mean(nodes, axis=0)), self.get_rotation(displacement)
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
