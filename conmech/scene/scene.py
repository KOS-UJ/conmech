from functools import partial
from typing import Callable, List, NamedTuple, Optional

import jax
import jax.numpy as jnp
import numba
import numpy as np
import torch

from conmech.dynamics.dynamics import DynamicsConfiguration, SolverMatrices
from conmech.dynamics.factory.dynamics_factory_method import ConstMatrices
from conmech.helpers import nph
from conmech.mesh.mesh import Mesh
from conmech.properties.body_properties import DynamicBodyProperties
from conmech.properties.mesh_properties import MeshProperties
from conmech.properties.obstacle_properties import ObstacleProperties
from conmech.properties.schedule import Schedule
from conmech.scene.body_forces import BodyForces
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
    lhs_acceleration_jax: np.ndarray
    rhs_acceleration: np.ndarray
    # lhs_acceleration_torch: np.ndarray
    # rhs_acceleration_torch: np.ndarray
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


def get_boundary_integral(
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

    penetration_norm = get_penetration_positive(
        displacement_step=boundary_displacement_step,
        normals=normals,
        penetration=args.penetration,
    )
    velocity_tangential = nph.get_tangential(boundary_v_new, normals)

    resistance_normal = obstacle_resistance_potential_normal(
        penetration_norm=penetration_norm, hardness=hardness, time_step=args.time_step
    )
    resistance_tangential = obstacle_resistance_potential_tangential(
        penetration_norm=args.penetration,
        tangential_velocity=velocity_tangential,
        friction=friction,
        time_step=args.time_step,
    )
    boundary_integral = (nodes_volume * (resistance_normal + resistance_tangential)).sum()
    return boundary_integral


def energy_vector_jax(value_vector, lhs, rhs):
    first = 0.5 * (lhs @ value_vector) - rhs
    value = first.reshape(-1) @ value_vector
    return value[0]


def energy_vector_torch(value_vector, lhs, rhs):
    first = 0.5 * (lhs @ value_vector) - rhs
    value = first.reshape(-1) @ value_vector
    return value[0]


def energy_obstacle(acceleration_vector, args: EnergyObstacleArguments):
    # print("Obstacle")
    # TODO: Repeat if collision
    main_energy0 = energy_vector_jax(
        value_vector=nph.stack_column(acceleration_vector),
        lhs=args.lhs_acceleration_jax,
        rhs=args.rhs_acceleration,
    )
    main_energy1 = compute_energy_jax(nph.unstack(acceleration_vector, dim=3), args)
    return main_energy0 + main_energy1


def energy_obstacle_torch(acceleration_vector, args: EnergyObstacleArguments):
    main_energy0 = energy_vector_torch(
        value_vector=nph.stack_column(acceleration_vector),
        lhs=args.lhs_acceleration_torch,
        rhs=args.rhs_acceleration_torch,
    )
    main_energy1 = compute_energy_jax(nph.unstack(acceleration_vector, dim=3), args)
    return main_energy0 + main_energy1


@jax.jit
def energy_obstacle_colliding_jax(acceleration_vector, args: EnergyObstacleArguments):
    # print("Obstacle colliding")
    # TODO: Repeat if collision
    main_energy = energy_obstacle(acceleration_vector, args)
    acceleration = nph.unstack(acceleration_vector, dim=3)
    boundary_integral = get_boundary_integral(acceleration=acceleration, args=args)
    return main_energy + boundary_integral


hes_energy_obstacle_new = jax.jit(
    lambda x, args: (lambda f, x, v: jax.grad(lambda x: jnp.vdot(jax.grad(f)(x, args), v))(x))(
        energy_obstacle, x, x
    )
)

hes_energy_obstacle_colliding_new = jax.jit(
    lambda x, args: (lambda f, x, v: jax.grad(lambda x: jnp.vdot(jax.grad(f)(x, args), v))(x))(
        energy_obstacle_colliding_jax, x, x
    )
)


energy_obstacle_jax = jax.jit(energy_obstacle)


@numba.njit
def get_closest_obstacle_to_boundary_numba(boundary_nodes, obstacle_nodes):
    boundary_obstacle_indices = np.zeros((len(boundary_nodes)), dtype=numba.int64)

    for i, boundary_node in enumerate(boundary_nodes):
        distances = nph.euclidean_norm_numba(obstacle_nodes - boundary_node)
        boundary_obstacle_indices[i] = distances.argmin()

    return boundary_obstacle_indices


###############


def get_jac(value, dx_big_jax):
    result0 = (
        (dx_big_jax @ jnp.tile(value, (3, 1))).reshape(3, -1, 3).swapaxes(0, 1).transpose((0, 2, 1))
    )
    return result0


def get_F_jax(value, dx_big_jax):
    I = jnp.eye(3)
    return get_jac(value, dx_big_jax) + I


def get_eps_lin_jax(F):
    I = jnp.eye(3)
    F_T = F.transpose((0, 2, 1))
    return 0.5 * (F + F_T) - I


def get_eps_rot_jax(F):
    I = jnp.eye(3)
    F_T = F.transpose((0, 2, 1))
    return 0.5 * (F_T @ F - I)


def compute_component_energy_jax(component, dx_big_jax, element_initial_volume, prop_1, prop_2):
    F_w = get_F_jax(component, dx_big_jax)
    eps_w = get_eps_rot_jax(F=F_w)  ## get_eps_lin_jax get_eps_rot_jax

    phi = prop_1 * (eps_w * eps_w).sum(axis=(1, 2)) + (prop_2 / 2.0) * (
        ((eps_w).trace(axis1=1, axis2=2) ** 2)
    )
    energy = element_initial_volume @ phi
    return energy


def compute_displacement_energy_jax(displacement, dx_big_jax, element_initial_volume, body_prop):
    return compute_component_energy_jax(
        component=displacement,
        dx_big_jax=dx_big_jax,
        element_initial_volume=element_initial_volume,
        prop_1=body_prop.mu,
        prop_2=body_prop.lambda_,
    )


def compute_velocity_energy_jax(velocity, dx_big_jax, element_initial_volume, body_prop):
    return compute_component_energy_jax(
        component=velocity,
        dx_big_jax=dx_big_jax,
        element_initial_volume=element_initial_volume,
        prop_1=body_prop.theta,
        prop_2=body_prop.zeta,
    )


def compute_energy_jax(acceleration, args):
    new_displacement = args.base_displacement + acceleration * args.time_step**2
    new_velocity = args.base_velocity + acceleration * args.time_step

    energy_new = (
        compute_displacement_energy_jax(
            displacement=new_displacement,
            dx_big_jax=args.dx_big_jax,
            element_initial_volume=args.element_initial_volume,
            body_prop=args.body_prop,
        )
        - args.base_energy_displacement
    ) / (args.time_step**2)

    energy_new += (
        compute_velocity_energy_jax(
            velocity=new_velocity,
            dx_big_jax=args.dx_big_jax,
            element_initial_volume=args.element_initial_volume,
            body_prop=args.body_prop,
        )
        - args.base_energy_velocity
    ) / (args.time_step)

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


#################


class Scene(BodyForces):
    def __init__(
        self,
        mesh_prop: MeshProperties,
        body_prop: DynamicBodyProperties,
        obstacle_prop: ObstacleProperties,
        schedule: Schedule,
        create_in_subprocess: bool,
        with_schur: bool = True,
    ):
        super().__init__(
            mesh_prop=mesh_prop,
            body_prop=body_prop,
            schedule=schedule,
            dynamics_config=DynamicsConfiguration(
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
                [BodyPosition(mesh_prop=mesh_prop, schedule=None) for mesh_prop in all_mesh_prop]
            )

    def get_energy_obstacle_args_for_jax(self, temperature=None):
        displacement = self.calculator_displacement_old
        velocity = self.calculator_velocity_old

        base_displacement = displacement + self.time_step * velocity
        body_prop = self.body_prop.get_tuple()

        args = EnergyObstacleArguments(
            lhs_acceleration_jax=self.solver_cache.lhs_acceleration_jax,
            rhs_acceleration=self.get_normalized_integrated_forces_column_for_jax(),
            #
            # lhs_acceleration_torch=torch.Tensor(
            #     np.array(self.solver_cache.lhs_acceleration_jax.todense())
            # ),
            # rhs_acceleration_torch=torch.Tensor(np.array(self.get_integrated_forces_column_jax())),
            # #
            boundary_velocity_old=None,  # jnp.asarray(self.norm_boundary_velocity_old),
            boundary_normals=None,  # jnp.asarray(self.get_normalized_boundary_normals()),
            boundary_obstacle_normals=None,  # jnp.asarray(self.get_norm_boundary_obstacle_normals()),
            penetration=None,  # jnp.asarray(self.get_penetration_scalar()),
            surface_per_boundary_node=None,  # jnp.asarray(self.get_surface_per_boundary_node()),
            body_prop=body_prop,
            obstacle_prop=self.obstacle_prop,
            time_step=self.time_step,
            base_displacement=base_displacement,
            element_initial_volume=self.matrices.element_initial_volume,
            dx_big_jax=self.matrices.dx_big_jax,
            base_energy_displacement=compute_displacement_energy_jax(
                displacement=base_displacement,
                dx_big_jax=self.matrices.dx_big_jax,
                element_initial_volume=self.matrices.element_initial_volume,
                body_prop=body_prop,
            ),
            base_velocity=velocity,
            base_energy_velocity=compute_velocity_energy_jax(
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

    def interpolate_base(self, scene):
        # self.inner_forces = scene.approximate_boundary_or_all_from_base(
        #     layer_number=1, base_values=scene.inner_forces
        # )
        # self.outer_forces = scene.approximate_boundary_or_all_from_base(
        #     layer_number=1, base_values=scene.outer_forces
        # )
        self.set_displacement_old(
            scene.approximate_boundary_or_all_from_base(
                layer_number=1, base_values=scene.displacement_old
            )
        )
        self.set_velocity_old(
            scene.approximate_boundary_or_all_from_base(
                layer_number=1, base_values=scene.velocity_old
            )
        )
