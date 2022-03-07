import time

import numpy as np
from conmech.helpers import nph
from deep_conmech.common import config
from deep_conmech.simulator.setting.setting_forces import *
from numba import njit


@njit
def obstacle_resistance_potential_normal(penetration):
    value = config.OBSTACLE_HARDNESS * 2.0 * penetration ** 2
    return (penetration > 0) * value * ((1.0 / config.TIMESTEP) ** 2)


@njit
def obstacle_resistance_potential_tangential(penetration, tangential_velocity):
    value = config.OBSTACLE_FRICTION * nph.euclidean_norm_numba(tangential_velocity)
    return (
        (penetration > 0) * value * ((1.0 / config.TIMESTEP))
    )  # rozkminic, do not use ReLU(normal_displacement)


def integrate(
    nodes,
    v,
    faces,
    face_normals,
    closest_to_faces_obstacle_normals,
    closest_to_faces_obstacle_origins,
):
    face_nodes = nodes[faces]

    middle_node = np.mean(face_nodes, axis=1)
    middle_node_penetration = (-1) * nph.elementwise_dot(
        middle_node - closest_to_faces_obstacle_origins,
        closest_to_faces_obstacle_normals,
    )

    face_v = v[faces]
    middle_v = np.mean(face_v, axis=1)
    middle_v_normal = nph.elementwise_dot(middle_v, face_normals, keepdims=True)

    middle_v_tangential = middle_v - (middle_v_normal * face_normals)

    edge_lengths = nph.euclidean_norm_numba(face_nodes[:, 0] - face_nodes[:, 1])
    resistance_normal = obstacle_resistance_potential_normal(middle_node_penetration)
    resistance_tangential = obstacle_resistance_potential_tangential(
        middle_node_penetration, middle_v_tangential
    )
    result = np.sum(edge_lengths * (resistance_normal + resistance_tangential))
    return result


@njit
def integrate_numba(
    nodes, nodes_normals, obstacle_nodes, obstacle_nodes_normals, v, nodes_volume
):
    result = 0.0
    for i in range(len(nodes_normals)):
        node_penetration = (
            (-1) * (nodes[i] - obstacle_nodes[i]) @ obstacle_nodes_normals[i]
        )

        node_v_normal = v[i] @ nodes_normals[i]
        node_v_tangential = v[i] - (node_v_normal * nodes_normals[i])

        resistance_normal = obstacle_resistance_potential_normal(node_penetration)
        resistance_tangential = obstacle_resistance_potential_tangential(
            node_penetration, node_v_tangential
        )

        result += nodes_volume[i] * (resistance_normal + resistance_tangential)
    return result


@njit
def get_closest_obstacle_to_boundary_numba(boundary_nodes, obstacle_nodes):
    boundary_obstacle_indices = np.zeros((len(boundary_nodes)), dtype=numba.int64)

    for i in range(len(boundary_nodes)):
        boundary_node = boundary_nodes[i]

        distances = nph.euclidean_norm_numba(obstacle_nodes - boundary_node)
        # valid_indices = np.where((obstacle_normals @ edge_normal) > 0)[0]
        # if(len(valid_indices) > 0):
        # min_index = valid_indices[distances[valid_indices].argmin()]

        boundary_obstacle_indices[i] = distances.argmin()

    return boundary_obstacle_indices


# @njit
def L2_obstacle_np(
    boundary_a,
    C_boundary,
    E_boundary,
    boundary_v_old,
    boundary_nodes,
    boundary_nodes_normals,
    boundary_obstacle_nodes,
    boundary_obstacle_normals,
    boundary_nodes_volume,
):
    value = L2_full_np(boundary_a, C_boundary, E_boundary)

    boundary_v_new = boundary_v_old + config.TIMESTEP * boundary_a
    boundary_nodes_new = boundary_nodes + config.TIMESTEP * boundary_v_new

    args = (
        boundary_nodes_new,
        boundary_nodes_normals,
        boundary_obstacle_nodes,
        boundary_obstacle_normals,
        boundary_v_new,
        boundary_nodes_volume,
    )
    boundary_integral = integrate_numba(*args)
    """
    boundary_integral = integrate(*args)
    """
    return value + boundary_integral


class SettingObstacle(SettingForces):
    def __init__(
        self,
        mesh_type,
        mesh_density_x,
        mesh_density_y,
        scale_x,
        scale_y,
        is_adaptive,
        create_in_subprocess,
    ):
        super().__init__(
            mesh_type,
            mesh_density_x,
            mesh_density_y,
            scale_x,
            scale_y,
            is_adaptive,
            create_in_subprocess,
        )
        self.obstacles = None
        self.clear()

    def prepare(self, forces):
        super().prepare(forces)
        self.boundary_obstacle_indices = get_closest_obstacle_to_boundary_numba(
            self.moved_points, self.obstacle_origins
        )

    def clear(self):
        super().clear()
        self.boundary_obstacle_nodes_indices = None

    def set_obstacles(self, obstacles_unnormalized):
        self.obstacles = obstacles_unnormalized
        self.obstacles[0, ...] = nph.normalize_euclidean_numba(self.obstacles[0, ...])

    @property
    def obstacle_normals(self):
        return self.obstacles[0, ...]

    @property
    def obstacle_origins(self):
        return self.obstacles[1, ...]

    @property
    def obstacle_nodes(self):
        return self.obstacles[1, ...]

    @property
    def obstacle_nodes_normals(self):
        return self.obstacles[0, ...]

    @property
    def boundary_obstacle_nodes(self):
        return self.obstacle_nodes[self.boundary_obstacle_indices]

    @property
    def boundary_obstacle_normals(self):
        return self.obstacle_normals[self.boundary_obstacle_indices]

    @property
    def normalized_boundary_obstacle_nodes(self):
        return self.normalize_rotate(
            self.boundary_obstacle_nodes - self.mean_moved_points
        )

    @property
    def normalized_boundary_obstacle_normals(self):
        return self.normalize_rotate(self.boundary_obstacle_normals)

    @property
    def normalized_obstacle_normals(self):
        return self.normalize_rotate(self.obstacle_normals)

    @property
    def normalized_obstacle_origins(self):
        return self.normalize_rotate(self.obstacle_origins - self.mean_moved_points)

    @property
    def obstacle_normal(self):
        return self.obstacle_normals[0]

    @property
    def obstacle_origin(self):
        return self.obstacle_origins[0]

    @property
    def normalized_obstacle_normal(self):
        return self.normalized_obstacle_normals[0]

    @property
    def normalized_obstacle_origin(self):
        return self.normalized_obstacle_origins[0]

    def normalized_L2_obstacle_np(self, normalized_boundary_a_vector):
        return L2_obstacle_np(
            nph.unstack(normalized_boundary_a_vector),
            self.C_boundary,
            self.normalized_E_boundary,
            self.normalized_boundary_v_old,
            self.normalized_boundary_points,
            self.normalized_boundary_nodes_normals,
            self.normalized_boundary_obstacle_nodes,
            self.normalized_boundary_obstacle_normals,
            self.boundary_nodes_volume,
        )

    @property
    def normalized_boundary_v_old(self):
        return self.normalized_v_old[: self.boundary_nodes_count, :]

    @property
    def normalized_boundary_points(self):
        return self.normalized_points[: self.boundary_nodes_count, :]

    @property
    def boundary_obstacle_penetration(self):
        node_penetration = (-1) * nph.elementwise_dot(
            self.moved_points - self.boundary_obstacle_nodes,
            self.boundary_obstacle_normals,
            keepdims=True,
        )
        return (node_penetration > 0) * node_penetration

    @property
    def boundary_obstacle_penetration_normals(self):
        return (
            self.boundary_obstacle_penetration
            * self.boundary_obstacle_normals
        )

    @property
    def normalized_boundary_obstacle_penetration_normals(self):
        return self.normalize_rotate(self.boundary_obstacle_penetration_normals)
