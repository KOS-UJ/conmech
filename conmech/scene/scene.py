from dataclasses import dataclass
from typing import List, Optional

import numba
import numpy as np

from conmech.dynamics.dynamics import DynamicsConfiguration, Dynamics
from conmech.helpers import nph
from conmech.mesh.boundaries_description import BoundariesDescription
from conmech.mesh.mesh import Mesh
from conmech.properties.mesh_properties import MeshProperties
from conmech.properties.obstacle_properties import ObstacleProperties
from conmech.scene.body_forces import energy
from conmech.simulations.problem_solver import Body
from conmech.state.body_position import BodyPosition
from conmech.state.state import State


def get_new_penetration_norm(displacement_step, normals, penetration):
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


@dataclass
class IntegrateArguments:
    velocity: np.ndarray
    displacement_step: np.ndarray
    penetration: np.ndarray
    normals: np.ndarray
    nodes_volume: np.ndarray
    hardness: float
    friction: float
    time_step: float


def integrate(args: IntegrateArguments):
    penetration_norm = get_new_penetration_norm(
        args.displacement_step, args.normals, args.penetration
    )
    velocity_tangential = nph.get_tangential(args.velocity, args.normals)

    resistance_normal = obstacle_resistance_potential_normal(
        penetration_norm, args.hardness, args.time_step
    )
    resistance_tangential = obstacle_resistance_potential_tangential(
        args.penetration, velocity_tangential, args.friction, args.time_step
    )
    result = (args.nodes_volume * (resistance_normal + resistance_tangential)).sum()
    return result


@dataclass
class EnergyObstacleArguments:
    lhs: np.ndarray
    rhs: np.ndarray
    boundary_velocity_old: np.ndarray
    boundary_normals: np.ndarray
    penetration: np.ndarray
    surface_per_boundary_node: np.ndarray
    obstacle_prop: ObstacleProperties
    time_step: float


def get_boundary_integral(acceleration, args: EnergyObstacleArguments):
    boundary_nodes_count = args.boundary_velocity_old.shape[0]
    boundary_a = acceleration[:boundary_nodes_count, :]  # TODO: boundary slice

    boundary_v_new = args.boundary_velocity_old + args.time_step * boundary_a
    boundary_displacement_step = args.time_step * boundary_v_new

    integrate_args = IntegrateArguments(
        velocity=boundary_v_new,
        displacement_step=boundary_displacement_step,
        penetration=args.penetration,
        normals=args.boundary_normals,
        nodes_volume=args.surface_per_boundary_node,
        hardness=args.obstacle_prop.hardness,
        friction=args.obstacle_prop.friction,
        time_step=args.time_step,
    )

    boundary_integral = integrate(integrate_args)
    return boundary_integral


def energy_obstacle(
    acceleration,
    args: EnergyObstacleArguments,
):
    main_energy = energy(acceleration, args.lhs, args.rhs)
    boundary_integral = get_boundary_integral(acceleration=acceleration, args=args)

    return main_energy + boundary_integral


@numba.njit
def get_closest_obstacle_to_boundary_numba(boundary_nodes, obstacle_nodes):
    boundary_obstacle_indices = np.zeros((len(boundary_nodes)), dtype=numba.int64)

    for i, boundary_node in enumerate(boundary_nodes):
        distances = nph.euclidean_norm_numba(obstacle_nodes - boundary_node)
        boundary_obstacle_indices[i] = distances.argmin()

    return boundary_obstacle_indices


class Scene:
    def __init__(
        self,
        body: "Body",
        time_step: float,
        obstacle_prop: ObstacleProperties,
        normalize_by_rotation: bool = False,
        create_in_subprocess: bool = False,
    ):
        self.body = body
        Dynamics(
            self.body, time_step, DynamicsConfiguration(create_in_subprocess=create_in_subprocess)
        )
        State(self.body)
        self.time_step = time_step
        self.obstacle_prop = obstacle_prop
        self.normalize_by_rotation = normalize_by_rotation
        self.create_in_subprocess = create_in_subprocess
        self.closest_obstacle_indices = None
        self.linear_obstacles: np.ndarray = np.array([[], []])
        self.mesh_obstacles: List[BodyPosition] = []

        self.body.dynamics.force.clear()

    def prepare(self, inner_forces):
        self.body.dynamics.force.prepare(inner_forces)
        if not self.has_no_obstacles:
            self.closest_obstacle_indices = get_closest_obstacle_to_boundary_numba(
                self.body.state.position.boundary_nodes, self.obstacle_nodes
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
            boundaries_description: ... = BoundariesDescription(
                contact=lambda x: True, dirichlet=lambda x: False
            )
            for mesh_prop in all_mesh_prop:
                mesh = Mesh(
                    mesh_prop=mesh_prop,
                    boundaries_description=boundaries_description,
                    create_in_subprocess=self.create_in_subprocess,
                )
                obstacle = Body(properties=None, mesh=mesh)
                self.mesh_obstacles.append(
                    BodyPosition(obstacle, normalize_by_rotation=self.normalize_by_rotation)
                )

    def get_normalized_energy_obstacle_np(self, temperature=None):
        (
            normalized_rhs_boundary,
            normalized_rhs_free,
        ) = self.body.dynamics.force.get_all_normalized_rhs_np(temperature)
        penetration = self.get_penetration()
        args = EnergyObstacleArguments(
            lhs=self.body.dynamics.solver_cache.lhs_boundary,  # TODO
            rhs=normalized_rhs_boundary,
            boundary_velocity_old=self.norm_boundary_velocity_old,
            boundary_normals=self.body.state.position.get_normalized_boundary_normals(),
            penetration=penetration,
            surface_per_boundary_node=self.body.state.position.get_surface_per_boundary_node(),
            obstacle_prop=self.obstacle_prop,
            time_step=self.time_step,
        )
        return (
            lambda normalized_boundary_a_vector: energy_obstacle(
                acceleration=nph.unstack(normalized_boundary_a_vector, self.body.mesh.dimension),
                args=args,
            ),
            normalized_rhs_free,
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

    def get_normalized_rhs_np(self, temperature=None):
        value = self.body.dynamics.force.get_normalized_rhs_np()
        return value

    def iterate_self(self, acceleration, temperature=None):
        return self.body.state.position.iterate_self(
            time_step=self.time_step, acceleration=acceleration
        )

    @property
    def norm_boundary_obstacle_nodes(self):
        return self.body.state.position.normalize_rotate(
            self.boundary_obstacle_nodes - self.body.state.position.mean_moved_nodes
        )

    def get_norm_obstacle_normals(self):
        return self.body.state.position.normalize_rotate(self.get_obstacle_normals())

    def get_boundary_obstacle_normals(self):
        return self.get_obstacle_normals()[self.closest_obstacle_indices]

    def get_norm_boundary_obstacle_normals(self):
        return self.body.state.position.normalize_rotate(self.get_boundary_obstacle_normals())

    @property
    def normalized_obstacle_nodes(self):
        return self.body.state.position.normalize_rotate(
            self.obstacle_nodes - self.mean_moved_nodes
        )

    @property
    def boundary_velocity_old(self):
        return self.body.state.position.velocity_old[self.body.mesh.boundary_indices]

    @property
    def boundary_a_old(self):
        return self.body.state.position.acceleration_old[self.body.mesh.boundary_indices]

    @property
    def norm_boundary_velocity_old(self):
        return self.body.state.position.rotated_velocity_old[self.body.mesh.boundary_indices]

    @property
    def normalized_boundary_nodes(self):
        return self.body.state.position.normalized_nodes[self.body.mesh.boundary_indices]

    def get_penetration(self):
        return (-1) * nph.elementwise_dot(
            (self.normalized_boundary_nodes - self.norm_boundary_obstacle_nodes),
            self.get_norm_boundary_obstacle_normals(),
        ).reshape(-1, 1)

    def get_penetration_norm(self):
        penetration = self.get_penetration()
        return penetration * (penetration > 0)

    def __get_boundary_penetration(self):
        return (-1) * self.get_penetration_norm() * self.body.state.position.get_boundary_normals()

    def get_normalized_boundary_penetration(self):
        return self.body.state.position.normalize_rotate(self.__get_boundary_penetration())

    def get_damping_input(self):
        return self.obstacle_prop.hardness * self.get_normalized_boundary_penetration()

    def __get_boundary_v_tangential(self):
        return nph.get_tangential(
            self.boundary_velocity_old, self.body.state.position.get_boundary_normals()
        )

    def __get_normalized_boundary_v_tangential(self):
        return nph.get_tangential(
            self.norm_boundary_velocity_old,
            self.body.state.position.get_normalized_boundary_normals(),
        )

    def get_friction_vector(self):
        return (self.get_penetration() > 0) * np.nan_to_num(
            nph.normalize_euclidean_numba(self.__get_normalized_boundary_v_tangential())
        )

    def get_friction_input(self):
        return self.obstacle_prop.friction * self.get_friction_vector()

    def get_resistance_normal(self):
        return obstacle_resistance_potential_normal(
            self.get_penetration_norm(), self.obstacle_prop.hardness, self.time_step
        )

    def get_resistance_tangential(self):
        return obstacle_resistance_potential_tangential(
            self.get_penetration_norm(),
            self.__get_boundary_v_tangential(),
            self.obstacle_prop.friction,
            self.time_step,
        )

    @staticmethod
    def complete_mesh_boundary_data_with_zeros(mesh: Mesh, data: np.ndarray):
        return np.pad(data, ((0, mesh.nodes_count - len(data)), (0, 0)), "constant")

    def complete_boundary_data_with_zeros(self, data: np.ndarray):
        return Scene.complete_mesh_boundary_data_with_zeros(self.body.mesh, data)

    @property
    def has_no_obstacles(self):
        return self.linear_obstacles.size == 0 and len(self.mesh_obstacles) == 0

    def get_colliding_nodes_indicator(self):
        if self.has_no_obstacles:
            return np.zeros((self.body.mesh.nodes_count, 1), dtype=np.int64)
        return self.complete_boundary_data_with_zeros((self.get_penetration() > 0) * 1)

    def is_colliding(self):
        return np.any(self.get_colliding_nodes_indicator())

    def get_colliding_all_nodes_indicator(self):
        if self.is_colliding():
            return np.ones((self.body.mesh.nodes_count, 1), dtype=np.int64)
        return np.zeros((self.body.mesh.nodes_count, 1), dtype=np.int64)

    def clear_for_save(self):
        self.element_initial_volume = None
        self.acceleration_operator = None
        self.thermal_expansion = None
        self.thermal_conductivity = None
