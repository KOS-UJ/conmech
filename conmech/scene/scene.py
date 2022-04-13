from dataclasses import dataclass
from typing import List, Optional

import numba
import numpy as np

from conmech.helpers import nph
from conmech.properties.body_properties import DynamicBodyProperties
from conmech.properties.mesh_properties import MeshProperties
from conmech.properties.obstacle_properties import ObstacleProperties
from conmech.properties.schedule import Schedule
from conmech.scene.setting_forces import SettingForces, energy
from conmech.state.body_position import BodyPosition


def get_penetration_norm(nodes, obstacle_nodes, obstacle_normals):
    projection = (-1) * nph.elementwise_dot((nodes - obstacle_nodes), obstacle_normals).reshape(
        -1, 1
    )
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
    nodes: np.ndarray
    normals: np.ndarray
    obstacle_nodes: np.ndarray
    obstacle_normals: np.ndarray
    nodes_volume: np.ndarray
    hardness: float
    friction: float
    time_step: float


def integrate(args: IntegrateArguments):
    penetration_norm = get_penetration_norm(args.nodes, args.obstacle_nodes, args.obstacle_normals)

    velocity_tangential = nph.get_tangential(args.velocity, args.normals)

    resistance_normal = obstacle_resistance_potential_normal(
        penetration_norm, args.hardness, args.time_step
    )
    resistance_tangential = obstacle_resistance_potential_tangential(
        penetration_norm, velocity_tangential, args.friction, args.time_step
    )
    result = (args.nodes_volume * (resistance_normal + resistance_tangential)).sum()
    return result


@dataclass
class EnergyObstacleArguments:
    lhs: np.ndarray
    rhs: np.ndarray
    boundary_velocity_old: np.ndarray
    boundary_nodes: np.ndarray
    boundary_normals: np.ndarray
    boundary_obstacle_nodes: np.ndarray
    boundary_obstacle_normals: np.ndarray
    surface_per_boundary_node: np.ndarray
    obstacle_prop: ObstacleProperties
    time_step: float


def energy_obstacle(
    acceleration: np.ndarray,
    args: EnergyObstacleArguments,
):
    value = energy(acceleration, args.lhs, args.rhs)

    boundary_nodes_count = args.boundary_velocity_old.shape[0]
    boundary_a = acceleration[:boundary_nodes_count, :]  # TODO: boundary slice

    boundary_v_new = args.boundary_velocity_old + args.time_step * boundary_a
    boundary_nodes_new = args.boundary_nodes + args.time_step * boundary_v_new

    integrate_args = IntegrateArguments(
        velocity=boundary_v_new,
        nodes=boundary_nodes_new,
        normals=args.boundary_normals,
        obstacle_nodes=args.boundary_obstacle_nodes,
        obstacle_normals=args.boundary_obstacle_normals,
        nodes_volume=args.surface_per_boundary_node,
        hardness=args.obstacle_prop.hardness,
        friction=args.obstacle_prop.friction,
        time_step=args.time_step,
    )

    boundary_integral = integrate(integrate_args)

    return value + boundary_integral


@numba.njit
def get_closest_obstacle_to_boundary_numba(boundary_nodes, obstacle_nodes):
    boundary_obstacle_indices = np.zeros((len(boundary_nodes)), dtype=numba.int64)

    for i, boundary_node in enumerate(boundary_nodes):
        distances = nph.euclidean_norm_numba(obstacle_nodes - boundary_node)
        boundary_obstacle_indices[i] = distances.argmin()

    return boundary_obstacle_indices


class Scene(SettingForces):
    def __init__(
        self,
        mesh_prop: MeshProperties,
        body_prop: DynamicBodyProperties,
        obstacle_prop: ObstacleProperties,
        schedule: Schedule,
        normalize_by_rotation: bool,
        create_in_subprocess,
    ):
        super().__init__(
            mesh_prop=mesh_prop,
            body_prop=body_prop,
            schedule=schedule,
            normalize_by_rotation=normalize_by_rotation,
            create_in_subprocess=create_in_subprocess,
        )
        self.obstacle_prop = obstacle_prop
        self.closest_obstacle_indices = None
        self.linear_obstacles: np.ndarray = np.array([[], []])
        self.mesh_obstacles: List[BodyPosition] = []

        self.clear()

    def prepare(self, forces):
        super().prepare(forces)
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

    def get_normalized_energy_obstacle_np(self, temperature=None):
        normalized_rhs_boundary, normalized_rhs_free = self.get_all_normalized_rhs_np(temperature)
        args = EnergyObstacleArguments(
            lhs=self.solver_cache.lhs_boundary,
            rhs=normalized_rhs_boundary,
            boundary_velocity_old=self.norm_boundary_velocity_old,
            boundary_nodes=self.normalized_boundary_nodes,
            boundary_normals=self.get_normalized_boundary_normals(),
            boundary_obstacle_nodes=self.norm_boundary_obstacle_nodes,
            boundary_obstacle_normals=self.get_norm_boundary_obstacle_normals(),
            surface_per_boundary_node=self.get_surface_per_boundary_node(),
            obstacle_prop=self.obstacle_prop,
            time_step=self.time_step,
        )
        return (
            lambda normalized_boundary_a_vector: energy_obstacle(
                acceleration=nph.unstack(normalized_boundary_a_vector, self.dimension), args=args
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

    def get_boundary_penetration_norm(self):
        return get_penetration_norm(
            self.boundary_nodes, self.boundary_obstacle_nodes, self.get_boundary_obstacle_normals()
        )

    def get_boundary_penetration(self):
        return self.get_boundary_penetration_norm() * self.get_boundary_obstacle_normals()

    def get_normalized_boundary_penetration(self):
        return self.normalize_rotate(self.get_boundary_penetration())

    def get_normalized_boundary_v_tangential(self):
        return nph.get_tangential(
            self.norm_boundary_velocity_old, self.get_normalized_boundary_normals()
        ) * (self.get_boundary_penetration_norm() > 0)

    def get_boundary_v_tangential(self):
        return nph.get_tangential(self.boundary_velocity_old, self.get_boundary_normals())

    def get_resistance_normal(self):
        return obstacle_resistance_potential_normal(
            self.get_boundary_penetration_norm(), self.obstacle_prop.hardness, self.time_step
        )

    def get_resistance_tangential(self):
        return obstacle_resistance_potential_tangential(
            self.get_boundary_penetration_norm(),
            self.get_boundary_v_tangential(),
            self.obstacle_prop.friction,
            self.time_step,
        )

    def complete_boundary_data_with_zeros(self, data):
        # return np.resize(data, (self.nodes_count, data.shape[1]))
        completed_data = np.zeros((self.nodes_count, data.shape[1]), dtype=data.dtype)
        completed_data[self.boundary_indices] = data
        return completed_data

    @property
    def has_no_obstacles(self):
        return self.linear_obstacles.size == 0 and len(self.mesh_obstacles) == 0

    def is_colliding(self):
        if self.has_no_obstacles:
            return False
        return np.any(self.get_boundary_penetration_norm() > 0)
