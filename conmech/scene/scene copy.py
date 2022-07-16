from typing import Callable, List, NamedTuple, Optional

import jax
import jax.numpy as jnp
import numba
import numpy as np

from conmech.dynamics.dynamics import DynamicsConfiguration, SolverMatrices
from conmech.dynamics.factory.dynamics_factory_method import ConstMatrices
from conmech.helpers import nph
from conmech.mesh.mesh import Mesh
from conmech.properties.body_properties import DynamicBodyProperties
from conmech.properties.mesh_properties import MeshProperties
from conmech.properties.obstacle_properties import ObstacleProperties
from conmech.properties.schedule import Schedule
from conmech.scene.body_forces import BodyForces, energy, energy_lhs
from conmech.state.body_position import BodyPosition


def get_penetration_positive(displacement_step, normals, penetration):
    projection = nph.elementwise_dot(displacement_step, normals, keepdims=True) + penetration
    return (projection > 0) * projection


def obstacle_resistance_potential_normal(penetration_norm, hardness, time_step):
    return hardness * 0.5 * (penetration_norm**2) * ((1.0 / time_step) ** 2)


def obstacle_resistance_potential_tangential(
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
    lhs_sparse_jax: np.ndarray
    rhs: np.ndarray
    boundary_velocity_old: np.ndarray
    boundary_normals: np.ndarray
    boundary_obstacle_normals: np.ndarray
    penetration: np.ndarray
    surface_per_boundary_node: np.ndarray
    body_prop: np.ndarray
    obstacle_prop: np.ndarray
    time_step: float
    colliding: bool
    w: np.ndarray
    element_initial_volume: np.ndarray
    dx_big_jax: np.ndarray
    energy_w: np.ndarray


def get_boundary_integral_internal(
    acceleration,
    args: EnergyObstacleArguments,
    get_resistance_normal: Callable,
    get_resistance_tangental: Callable,
):
    boundary_nodes_count = args.boundary_velocity_old.shape[0]
    boundary_a = acceleration[:boundary_nodes_count, :]  # TODO: boundary slice

    boundary_v_new = args.boundary_velocity_old + args.time_step * boundary_a
    boundary_displacement_step = args.time_step * boundary_v_new

    normals = args.boundary_normals
    nodes_volume = args.surface_per_boundary_node
    hardness = args.obstacle_prop.hardness
    friction = args.obstacle_prop.friction

    penetration_norm = get_penetration_positive(
        displacement_step=boundary_displacement_step,
        normals=normals,
        penetration=args.penetration,
    )
    velocity_tangential = nph.get_tangential(boundary_v_new, normals)

    resistance_normal = get_resistance_normal(
        penetration_norm=penetration_norm, hardness=hardness, time_step=args.time_step
    )
    resistance_tangential = get_resistance_tangental(
        penetration_norm=args.penetration,
        tangential_velocity=velocity_tangential,
        friction=friction,
        time_step=args.time_step,
    )
    boundary_integral = (nodes_volume * (resistance_normal + resistance_tangential)).sum()
    return boundary_integral


def get_boundary_integral(acceleration, args: EnergyObstacleArguments):
    return get_boundary_integral_internal(
        acceleration=acceleration,
        args=args,
        get_resistance_normal=obstacle_resistance_potential_normal,
        get_resistance_tangental=obstacle_resistance_potential_tangential,
    )


def get_boundary_function_integral(acceleration, args: EnergyObstacleArguments):
    return get_boundary_integral_internal(
        acceleration=acceleration,
        args=args,
        get_resistance_normal=obstacle_resistance_normal,
        get_resistance_tangental=obstacle_resistance_tangential,
    )


def energy_obstacle(
    acceleration,
    args: EnergyObstacleArguments,
):
    main_energy = energy(acceleration, args.solver_cache, args.rhs)
    boundary_integral = get_boundary_integral(acceleration=acceleration, args=args)
    return main_energy + boundary_integral


def energy_obstacle_U(displacement, setting, args: dict, colliding: bool):
    # TODO: Repeat if collision
    main_energy0 = energy_lhs(
        value=displacement, lhs=args.solver_cache.lhs_sparse_U_jax, rhs=args.rhs
    )

    main_energy1 = setting.compute_energy_U(displacement)
    return main_energy0 + main_energy1


def energy_obstacle_new(acceleration_vector, args: EnergyObstacleArguments):
    # TODO: Repeat if collision
    acceleration = nph.unstack(acceleration_vector, dim=3)
    main_energy0 = energy_lhs(value=acceleration, lhs=args.lhs_sparse_jax, rhs=args.rhs)
    main_energy1 = compute_energy(acceleration, args)
    return main_energy0 + main_energy1


energy_obstacle_new_jax = jax.jit(energy_obstacle_new)


def energy_obstacle_colliding_new(acceleration_vector, args: EnergyObstacleArguments):
    # TODO: Repeat if collision
    main_energy = energy_obstacle_new(acceleration_vector, args)
    acceleration = nph.unstack(acceleration_vector, dim=3)
    boundary_integral = get_boundary_integral(acceleration=acceleration, args=args)
    return main_energy + boundary_integral


@numba.njit
def get_closest_obstacle_to_boundary_numba(boundary_nodes, obstacle_nodes):
    boundary_obstacle_indices = np.zeros((len(boundary_nodes)), dtype=numba.int64)

    for i, boundary_node in enumerate(boundary_nodes):
        distances = nph.euclidean_norm_numba(obstacle_nodes - boundary_node)
        boundary_obstacle_indices[i] = distances.argmin()

    return boundary_obstacle_indices


###############
I = jnp.eye(3)


def get_jac(value, dx_big_jax):
    result0 = (
        (dx_big_jax @ jnp.tile(value, (3, 1))).reshape(3, -1, 3).swapaxes(0, 1).transpose((0, 2, 1))
    )
    return result0


def get_F(value, dx_big_jax, I):
    return get_jac(value, dx_big_jax) + I


def get_eps_lin(F, I):
    F_T = F.transpose((0, 2, 1))
    return 0.5 * (F + F_T) - I


def get_eps_rot(F, I):
    F_T = F.transpose((0, 2, 1))
    return 0.5 * (F_T @ F - I)


def compute_energy_U(displacement, dx_big_jax, element_initial_volume, body_prop):
    # dimension = displacement.shape[-1]

    F_u = get_F(displacement, dx_big_jax, I)
    eps_u = get_eps_rot(F=F_u, I=I)  # get_eps_lin get_eps_rot

    phi = body_prop.mu * (eps_u * eps_u).sum(axis=(1, 2)) + (body_prop.lambda_ / 2.0) * (
        ((eps_u).trace(axis1=1, axis2=2) ** 2)
    )
    energy1 = element_initial_volume @ phi
    return energy1


def compute_energy(acceleration, args):
    new_displacement = args.w + acceleration * args.time_step**2

    energy_new = (
        compute_energy_U(
            new_displacement,
            dx_big_jax=args.dx_big_jax,
            element_initial_volume=args.element_initial_volume,
            body_prop=args.body_prop,
        )
        - args.energy_w
    ) / (args.time_step**2)
    return energy_new

    dimension = acceleration.shape[-1]

    new_velocity_half = self.velocity_old + self.time_step * acceleration * 0.5
    new_displacement_half = self.displacement_old + self.time_step * new_velocity_half

    # new_nodes = self.initial_nodes + new_displacement
    # element_nodes = new_nodes[self.elements].transpose(1, 2, 0)
    # D_s = jnp.dstack(
    #     (
    #         element_nodes[0] - element_nodes[3],
    #         element_nodes[1] - element_nodes[3],
    #         element_nodes[2] - element_nodes[3],
    #     )
    # ).transpose(1, 0, 2)
    # F_u = D_s @ self.B_m

    I = jnp.eye(dimension)
    F_u_half = self.get_F(new_displacement_half, I)
    eps_u_half = self.get_eps_lin(F=F_u_half, I=I)  # get_eps_lin get_eps_rot

    jac_a = self.get_jac(acceleration)

    # phi = self.body_prop.mu * (eps_u * eps_u).sum(axis=(1, 2)) + (
    #     self.body_prop.lambda_ / 2.0
    # ) * (((eps_u).trace(axis1=1, axis2=2) ** 2))
    # energy1a = self.matrices.element_initial_volume @ phi
    # energy1 = energy1a * 1 / (self.time_step**2)

    ########
    P1 = 2 * self.body_prop.mu * eps_u_half + self.body_prop.lambda_ * (
        (eps_u_half).trace(axis1=1, axis2=2).repeat(dimension).reshape(-1, dimension, 1) * I
    )
    phi = (P1 * jac_a).sum(axis=(1, 2))
    # phi = ((F_u @ P1) * jac_a).sum(axis=(1, 2)
    # instead of eps_a (for symetric sigma - the same)
    energy2 = self.matrices.element_initial_volume @ phi


# ################


class Scene(BodyForces):
    def __init__(
        self,
        mesh_prop: MeshProperties,
        body_prop: DynamicBodyProperties,
        obstacle_prop: ObstacleProperties,
        schedule: Schedule,
        normalize_by_rotation: bool,
        create_in_subprocess: bool,
        with_schur: bool = True,
    ):
        super().__init__(
            mesh_prop=mesh_prop,
            body_prop=body_prop,
            schedule=schedule,
            dynamics_config=DynamicsConfiguration(
                normalize_by_rotation=normalize_by_rotation,
                create_in_subprocess=create_in_subprocess,
                with_lhs=True,
                with_schur=with_schur,
            ),
        )
        self.obstacle_prop = obstacle_prop
        self.closest_obstacle_indices = None
        self.linear_obstacles: np.ndarray = np.array([[], []])
        self.mesh_obstacles: List[BodyPosition] = []

        self.clear()

    def prepare(self, inner_forces):
        super().prepare(inner_forces)
        if not self.has_no_obstacles:
            self.closest_obstacle_indices = get_closest_obstacle_to_boundary_numba(
                self.boundary_nodes, self.obstacle_nodes
            )

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
                [
                    BodyPosition(mesh_prop=mesh_prop, schedule=None, normalize_by_rotation=False)
                    for mesh_prop in all_mesh_prop
                ]
            )

    def get_normalized_energy_obstacle_jax_U(self, temperature=None):
        normalized_rhs = jnp.asarray(self.get_normalized_rhs_cp_U(temperature).get())

        args = EnergyObstacleArguments(
            solver_cache=self.solver_cache,
            rhs=normalized_rhs,
            boundary_velocity_old=jnp.asarray(self.norm_boundary_velocity_old),
            boundary_normals=jnp.asarray(self.get_normalized_boundary_normals()),
            boundary_obstacle_normals=jnp.asarray(self.get_norm_boundary_obstacle_normals()),
            penetration=jnp.asarray(self.get_penetration_scalar()),
            surface_per_boundary_node=jnp.asarray(self.get_surface_per_boundary_node()),
            obstacle_prop=self.obstacle_prop,
            time_step=self.time_step,
        )
        colliding = self.is_colliding()
        return (
            lambda normalized_u_vector: energy_obstacle_U(
                displacement=nph.unstack(normalized_u_vector, self.dimension),
                setting=self,
                args=args,
                colliding=colliding,
            ),
            None,
        )

    def get_normalized_energy_obstacle_jax_new(self, temperature=None):
        w = self.displacement_old + self.time_step * self.velocity_old
        body_prop = self.body_prop.get_tuple()

        args = EnergyObstacleArguments(
            lhs_sparse_jax=self.solver_cache.lhs_sparse_jax,
            rhs=jnp.asarray(self.get_normalized_rhs_cp(temperature).get()),
            boundary_velocity_old=jnp.asarray(self.norm_boundary_velocity_old),
            boundary_normals=jnp.asarray(self.get_normalized_boundary_normals()),
            boundary_obstacle_normals=jnp.asarray(self.get_norm_boundary_obstacle_normals()),
            penetration=jnp.asarray(self.get_penetration_scalar()),
            surface_per_boundary_node=jnp.asarray(self.get_surface_per_boundary_node()),
            body_prop=body_prop,
            obstacle_prop=self.obstacle_prop,
            time_step=self.time_step,
            colliding=self.is_colliding(),
            w=self.displacement_old + self.time_step * self.velocity_old,
            element_initial_volume=self.matrices.element_initial_volume,
            dx_big_jax=self.matrices.dx_big_jax,
            energy_w=compute_energy_U(
                w,
                self.matrices.dx_big_jax,
                self.matrices.element_initial_volume,
                body_prop,
            ),
        )
        return args

    def get_normalized_energy_obstacle_jax(self, temperature=None):
        normalized_rhs_boundary, normalized_rhs_free = jnp.array(
            self.get_all_normalized_rhs_jax(temperature)
        )
        # normalized_rhs = jnp.asarray(self.get_normalized_rhs_cp(temperature).get())

        args = EnergyObstacleArguments(
            solver_cache=self.solver_cache,
            rhs=normalized_rhs_boundary,  # normalized_rhs,  # normalized_rhs_boundary,
            boundary_velocity_old=jnp.asarray(self.norm_boundary_velocity_old),
            boundary_normals=jnp.asarray(self.get_normalized_boundary_normals()),
            boundary_obstacle_normals=jnp.asarray(self.get_norm_boundary_obstacle_normals()),
            penetration=jnp.asarray(self.get_penetration_scalar()),
            surface_per_boundary_node=jnp.asarray(self.get_surface_per_boundary_node()),
            obstacle_prop=self.obstacle_prop,
            time_step=self.time_step,
        )
        return (
            lambda normalized_boundary_a_vector: energy_obstacle(
                acceleration=nph.unstack(normalized_boundary_a_vector, self.dimension), args=args
            ),
            normalized_rhs_free,  # None,  # normalized_rhs_free,
        )

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
    def boundary_a_old(self):
        return self.acceleration_old[self.boundary_indices]

    @property
    def norm_boundary_velocity_old(self):
        return self.rotated_velocity_old[self.boundary_indices]

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
        return nph.get_tangential(
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
        return obstacle_resistance_potential_normal(
            self.get_penetration_positive(), self.obstacle_prop.hardness, self.time_step
        )

    def get_resistance_tangential(self):
        return obstacle_resistance_potential_tangential(
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

    def get_colliding_nodes_indicator(self):
        if self.has_no_obstacles:
            return np.zeros((self.nodes_count, 1), dtype=np.int64)
        return self.complete_boundary_data_with_zeros((self.get_penetration_scalar() > 0) * 1)

    def is_colliding(self):
        return np.any(self.get_colliding_nodes_indicator())

    def get_colliding_all_nodes_indicator(self):
        if self.is_colliding():
            return np.ones((self.nodes_count, 1), dtype=np.int64)
        return np.zeros((self.nodes_count, 1), dtype=np.int64)

    def prepare_to_save(self):
        self.matrices = ConstMatrices()
        # lhs_sparse = self.solver_cache.lhs_sparse
        self.solver_cache = SolverMatrices()
        # self.solver_cache.lhs_sparse = lhs_sparse
