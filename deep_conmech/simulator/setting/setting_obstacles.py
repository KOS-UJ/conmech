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
    normals,
    closest_to_faces_obstacle_normals,
    closest_to_faces_obstacle_origins,
):
    edge_node = nodes[faces]
    edge_v = v[faces]

    middle_node = np.mean(edge_node, axis=1)
    middle_v = np.mean(edge_v, axis=1)

    middle_node_penetration = (-1) * nph.elementwise_dot(
        middle_node - closest_to_faces_obstacle_origins, closest_to_faces_obstacle_normals
    )
    middle_v_normal = nph.elementwise_dot(middle_v, normals, keepdims=True)

    middle_v_tangential = middle_v - (middle_v_normal * normals)

    edge_lengths = nph.euclidean_norm_numba(edge_node[:, 0] - edge_node[:, 1])
    resistance_normal = obstacle_resistance_potential_normal(middle_node_penetration)
    resistance_tangential = obstacle_resistance_potential_tangential(
        middle_node_penetration, middle_v_tangential
    )
    result = np.sum(edge_lengths * (resistance_normal + resistance_tangential))
    return result


@njit
def integrate_numba(nodes, v, faces, normals,
    closest_to_faces_obstacle_normals, closest_to_faces_obstacle_origins):
    result = 0.0
    for i in range(len(faces)):
        normal = normals[i] #(normal to obstacle and not edge?)
        obstacle_normal = closest_to_faces_obstacle_normals[i]
        origin = closest_to_faces_obstacle_origins[i]
        # if(np.sum(obstacle_normal) == 0):
        #    continue
        f1, f2 = faces[i]
        middle_node = 0.5 * (nodes[f1] + nodes[f2])
        middle_v = 0.5 * (v[f1] + v[f2])

        middle_node_penetration = (-1) * obstacle_normal @ (middle_node - origin)
        middle_v_normal = normal @ middle_v
        middle_v_tangential = middle_v - (middle_v_normal * normal)

        resistance_normal = obstacle_resistance_potential_normal(middle_node_penetration)
        resistance_tangential = obstacle_resistance_potential_tangential(
            middle_node_penetration, middle_v_tangential
        )

        edge_length = nph.euclidean_norm_numba(
            nodes[f1] - nodes[f2]
        )  # integrate with initial edge lengths?
        result += edge_length * (resistance_normal + resistance_tangential)
    return result


@njit
def get_closest_to_faces_obstacle_data(
    nodes, normals, faces, obstacle_normals, obstacle_origins
):
    closest_to_faces_obstacle_origins = np.zeros((len(faces), config.DIM))
    closest_to_faces_obstacle_normals = np.zeros((len(faces), config.DIM))

    for i in range(len(faces)):
        f1, f2 = faces[i]
        middle_node = 0.5 * (nodes[f1] + nodes[f2])
        # edge_normal = normals[i]

        distances = nph.euclidean_norm_numba(obstacle_origins - middle_node)
        # valid_indices = np.where((obstacle_normals @ edge_normal) > 0)[0]
        # if(len(valid_indices) > 0):
        # min_index = valid_indices[distances[valid_indices].argmin()]
        min_index = distances.argmin()

        closest_to_faces_obstacle_origins[i] = obstacle_origins[min_index]
        closest_to_faces_obstacle_normals[i] = obstacle_normals[min_index]

    return closest_to_faces_obstacle_origins, closest_to_faces_obstacle_normals


# @njit
def L2_obstacle_np(
    boundary_a,
    C_boundary,
    E_boundary,
    boundary_v_old,
    boundary_points,
    boundary_faces,
    boundary_normals,
    closest_to_faces_obstacle_normals,
    closest_to_faces_obstacle_origins,
):
    value = L2_full_np(boundary_a, C_boundary, E_boundary)

    boundary_v_new = boundary_v_old + config.TIMESTEP * boundary_a
    boundary_points_new = boundary_points + config.TIMESTEP * boundary_v_new
    
    args = (
        boundary_points_new,
        boundary_v_new,
        boundary_faces,
        boundary_normals,
        closest_to_faces_obstacle_normals,
        closest_to_faces_obstacle_origins
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
        self.set_empty_closest_to_faces_obstacle_data()

    def set_empty_closest_to_faces_obstacle_data(self):
        self.closest_to_faces_obstacle_origins = None
        self.closest_to_faces_obstacle_normals = None

    def prepare(self, forces):
        super().prepare(forces)
        self.set_closest_to_faces_obstacle_data()

    def clear(self):
        super().clear()
        self.set_empty_closest_to_faces_obstacle_data()

    def set_obstacles(self, obstacles_unnormalized):
        self.obstacles = obstacles_unnormalized
        self.obstacles[0, ...] = nph.normalize_euclidean_numba(self.obstacles[0, ...])

    def set_closest_to_faces_obstacle_data(self):
        (
            self.closest_to_faces_obstacle_origins,
            self.closest_to_faces_obstacle_normals,
        ) = get_closest_to_faces_obstacle_data(
            self.moved_points,
            self.boundary_normals,
            self.boundary_faces,
            self.obstacle_normals,
            self.obstacle_origins,
        )

    @property
    def obstacle_normals(self):
        return self.obstacles[0, ...]

    @property
    def obstacle_origins(self):
        return self.obstacles[1, ...]

    @property
    def normalized_closest_to_faces_obstacle_normals(self):
        return self.normalize_rotate(self.closest_to_faces_obstacle_normals)

    @property
    def normalized_closest_to_faces_obstacle_origins(self):
        return self.normalize_rotate(
            self.closest_to_faces_obstacle_origins - self.mean_moved_points
        )

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
            self.boundary_faces,
            self.boundary_normals,
            self.normalized_closest_to_faces_obstacle_normals,
            self.normalized_closest_to_faces_obstacle_origins,
        )

    @property
    def normalized_boundary_v_old(self):
        return self.normalized_v_old[: self.boundary_nodes_count, :]

    @property
    def normalized_boundary_points(self):
        return self.normalized_points[: self.boundary_nodes_count, :]

    @property
    def boundary_centers_penetration_scale(self):
        normals = -self.closest_to_faces_obstacle_normals

        boundary_centers_to_obstacle_at_normal = nph.elementwise_dot(
            self.boundary_centers - self.closest_to_faces_obstacle_origins,
            normals,
            keepdims=True,
        )

        return (
            boundary_centers_to_obstacle_at_normal > 0
        ) * boundary_centers_to_obstacle_at_normal

    @property
    def boundary_centers_penetration(self):
        return (
            self.boundary_centers_penetration_scale
            * self.closest_to_faces_obstacle_normals
        )

    @property
    def normalized_boundary_centers_penetration(self):
        return self.normalize_rotate(self.boundary_centers_penetration)
