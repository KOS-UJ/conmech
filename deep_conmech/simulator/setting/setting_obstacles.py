import numpy as np
from conmech.helpers import nph
from deep_conmech.common import config
from deep_conmech.simulator.setting.setting_forces import *
from numba import njit


####
def get_node_penetration(nodes, obstacle_nodes, obstacle_nodes_normals):
    return (-1) * nph.elementwise_dot((nodes - obstacle_nodes), obstacle_nodes_normals)

@njit
def get_node_penetration_numba(nodes, obstacle_nodes, obstacle_nodes_normals):
    return (-1) * ((nodes - obstacle_nodes) @ obstacle_nodes_normals)

def get_tangential(vector, normal):
    return vector - (nph.elementwise_dot(vector, normal, keepdims=True) * normal)

@njit
def get_tangential_numba(vector, normal):
    return vector - ((vector @ normal) * normal)


####

def obstacle_resistance_potential_normal(penetration):
    value = config.OBSTACLE_HARDNESS * 0.5 * (penetration ** 2)
    return (penetration > 0) * value * ((1.0 / config.TIMESTEP) ** 2)

obstacle_resistance_potential_normal_numba = numba.njit(
    obstacle_resistance_potential_normal
)


def obstacle_resistance_potential_tangential_internal(
    penetration, tangential_velocity, norm=nph.euclidean_norm_numba
):
    return (
        (penetration > 0)
        * config.OBSTACLE_FRICTION
        * norm(tangential_velocity)
        * (1.0 / config.TIMESTEP)
    )


def obstacle_resistance_potential_tangential(penetration, tangential_velocity):
    return obstacle_resistance_potential_tangential_internal(
        penetration, tangential_velocity, norm=nph.euclidean_norm
    )


obstacle_resistance_potential_tangential_numba = numba.njit(
    obstacle_resistance_potential_tangential_internal
)


def integrate(
    nodes,
    nodes_old,
    nodes_normals,
    obstacle_nodes,
    obstacle_nodes_normals,
    v,
    nodes_volume,
):

    node_penetration = get_node_penetration(
        nodes, obstacle_nodes, obstacle_nodes_normals
    )
    node_penetration_old = get_node_penetration(
        nodes_old, obstacle_nodes, obstacle_nodes_normals,
    )

    node_v_tangential = get_tangential_numba(v, obstacle_nodes_normals)

    resistance_normal = obstacle_resistance_potential_normal(node_penetration)
    resistance_tangential = obstacle_resistance_potential_tangential(
        node_penetration_old, node_v_tangential
    )
    result = (nodes_volume * (resistance_normal + resistance_tangential)).sum()
    return result


@njit
def integrate_numba(
    nodes,
    nodes_old,
    nodes_normals,
    obstacle_nodes,
    obstacle_nodes_normals,
    v,
    nodes_volume,
):
    result = 0.0
    for i in range(len(nodes)):
        node_penetration = get_node_penetration_numba(
            nodes[i], obstacle_nodes[i], obstacle_nodes_normals[i]
        )
        node_penetration_old = get_node_penetration_numba(
            nodes_old[i], obstacle_nodes[i], obstacle_nodes_normals[i]
        )

        node_v_tangential = get_tangential_numba(v[i], obstacle_nodes_normals[i])

        resistance_normal = obstacle_resistance_potential_normal_numba(node_penetration)
        resistance_tangential = obstacle_resistance_potential_tangential_numba(
            node_penetration_old, node_v_tangential
        )

        result += nodes_volume[i].item() * (resistance_normal + resistance_tangential)
    return result


@njit
def get_closest_obstacle_to_boundary_numba(boundary_nodes, obstacle_nodes):
    boundary_obstacle_indices = np.zeros((len(boundary_nodes)), dtype=numba.int64)

    for i in range(len(boundary_nodes)):
        distances = nph.euclidean_norm_numba(obstacle_nodes - boundary_nodes[i])
        boundary_obstacle_indices[i] = distances.argmin()

    return boundary_obstacle_indices


def L2_obstacle(
    a,
    C,
    E,
    boundary_v_old,
    boundary_nodes,
    boundary_normals,
    boundary_obstacle_nodes,
    boundary_obstacle_normals,
    boundary_nodes_volume,
):
    value = L2_new(a, C, E)

    boundary_nodes_count = boundary_v_old.shape[0]
    boundary_a = a[:boundary_nodes_count, :]  # TODO: boundary slice

    boundary_v_new = boundary_v_old + config.TIMESTEP * boundary_a
    boundary_nodes_new = boundary_nodes + config.TIMESTEP * boundary_v_new

    args = (
        boundary_nodes_new,
        boundary_nodes,
        boundary_normals,
        boundary_obstacle_nodes,
        boundary_obstacle_normals,
        boundary_v_new,
        boundary_nodes_volume,
    )

    is_numpy = isinstance(a, np.ndarray)
    boundary_integral = integrate_numba(*args) if is_numpy else integrate(*args)
    return value + boundary_integral


class SettingObstacles(SettingForces):
    def __init__(
        self,
        mesh_type,
        mesh_density_x,
        mesh_density_y=None,
        scale_x=None,
        scale_y=None,
        is_adaptive=False,
        create_in_subprocess=False,
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

    @property
    def boundary_penetration(self):
        return (-1) * nph.elementwise_dot(
            self.boundary_nodes - self.boundary_obstacle_nodes,
            self.boundary_obstacle_normals,
        )

    @property
    def resistance_normal(self):
        resistance_normal = obstacle_resistance_potential_normal(
            self.boundary_penetration
        )
        return resistance_normal.reshape(-1, 1)

    @property
    def normalized_boundary_v_tangential(self):
        return self.normalize_rotate(self.boundary_v_tangential)

    @property
    def boundary_v_tangential(self):
        return get_tangential(self.boundary_v_old, self.obstacle_nodes_normals)

    @property
    def resistance_tangential(self):
        resistance_tangential = obstacle_resistance_potential_tangential(
            self.boundary_penetration, self.boundary_v_tangential
        )
        return resistance_tangential.reshape(-1, 1)

    def prepare(self, forces):
        super().prepare(forces)
        self.boundary_obstacle_indices = get_closest_obstacle_to_boundary_numba(
            self.boundary_nodes, self.obstacle_origins
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
            self.boundary_obstacle_nodes - self.mean_moved_nodes
        )

    @property
    def normalized_boundary_obstacle_normals(self):
        return self.normalize_rotate(self.boundary_obstacle_normals)

    @property
    def normalized_obstacle_normals(self):
        return self.normalize_rotate(self.obstacle_normals)

    @property
    def normalized_obstacle_origins(self):
        return self.normalize_rotate(self.obstacle_origins - self.mean_moved_nodes)

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

    def get_normalized_L2_obstacle_np(self):
        return lambda normalized_boundary_a_vector: L2_obstacle(
            nph.unstack(normalized_boundary_a_vector, self.dim),
            self.C_boundary,
            self.normalized_E_boundary,
            self.normalized_boundary_v_old,
            self.normalized_boundary_nodes,
            self.normalized_boundary_normals,
            self.normalized_boundary_obstacle_nodes,
            self.normalized_boundary_obstacle_normals,
            self.boundary_nodes_volume,
        )

    @property
    def boundary_v_old(self):
        return self.v_old[: self.boundary_nodes_count, :]

    @property
    def normalized_boundary_v_old(self):
        return self.normalized_v_old[: self.boundary_nodes_count, :]

    @property
    def normalized_boundary_nodes(self):
        return self.normalized_points[: self.boundary_nodes_count, :]

    @property
    def boundary_obstacle_penetration(self):
        node_penetration = (-1) * nph.elementwise_dot(
            self.boundary_nodes - self.boundary_obstacle_nodes,
            self.boundary_obstacle_normals,
            keepdims=True,
        )
        return (node_penetration > 0) * node_penetration

    @property
    def boundary_obstacle_penetration_vectors(self):
        return self.boundary_obstacle_penetration * self.boundary_obstacle_normals

    @property
    def normalized_boundary_obstacle_penetration_vectors(self):
        return self.normalize_rotate(self.boundary_obstacle_penetration_vectors)
