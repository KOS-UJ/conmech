from ctypes import ArgumentError

import numpy as np
from numba import cuda, jit, njit, prange

import config
import helpers
import mesh_builders

# import os, sys
# sys.path.append(os.path.abspath('../'))


@njit
def get_edges_list(edges_matrix, edges):
    p = len(edges_matrix[0])
    e = 0
    for i in range(p):
        for j in range(p):
            if edges_matrix[i, j]:
                edges[e] = np.array([i, j])
                e += 1


@njit  # (parallel=True)
def get_edges_matrix(cells, edges_matrix):
    vertices_number = len(cells[0])
    for cell_index in range(len(cells)):  # TODO: prange?
        cell = cells[cell_index]
        for i in range(vertices_number):
            for j in range(vertices_number):
                edges_matrix[cell[i], cell[j]] = 1.0


##############


class Mesh:
    def __init__(self, mesh_size, mesh_type, corners, is_adaptive):
        self.mesh_size = mesh_size
        self.mesh_type = mesh_type
        self.corners = corners
        self.is_adaptive = is_adaptive
        self.min = helpers.min(corners)
        self.max = helpers.max(corners)

        if mesh_type == "cross":
            self.initial_points, self.cells = mesh_builders.get_cross_rectangle(
                self.mesh_size, self.corners
            )
        elif mesh_type == "meshzoo":
            self.initial_points, self.cells = mesh_builders.get_meshzoo_rectangle(
                self.mesh_size, self.corners
            )
        elif mesh_type == "dmsh":
            self.initial_points, self.cells = mesh_builders.get_dmsh_rectangle(
                self.mesh_size, self.corners
            )
        elif mesh_type == "pygmsh":
            self.initial_points, self.cells = mesh_builders.get_pygmsh_rectangle(
                self.mesh_size, self.corners, self.is_adaptive
            )
        else:
            raise ArgumentError

        # self.labels = self.get_labels()

        self.set_u_old(np.zeros_like(self.initial_points))

        self.initial_left_bottom = np.array(self.min)
        self.initial_right_top = np.array(self.max)

        self.corner_indices = self.get_corner_indices(
            self.initial_left_bottom, self.initial_right_top
        )
        self.point_types = self.get_point_types(
            self.points_number,
            self.initial_points,
            self.corner_indices,
            self.initial_left_bottom,
            self.initial_right_top,
        )


        self.moved_points2 = np.zeros_like(self.initial_points)
        self.normalized_initial_points = self.normalized_points.copy()

        self.set_v_old(np.zeros_like(self.initial_points))
        self.set_a_old(np.zeros_like(self.initial_points))

        p = self.points_number
        self.edges_matrix = np.zeros((p, p), dtype=np.bool)
        get_edges_matrix(self.cells, self.edges_matrix)

        e = np.sum(self.edges_matrix)
        self.edges = np.zeros((e, 2), dtype=np.int)
        get_edges_list(self.edges_matrix, self.edges)

        self.on_gamma_d = (
            self.initial_points[:, 0] == self.initial_left_bottom[0]
        ).reshape(-1, 1)
        self.on_gamma_d_stack = np.vstack((self.on_gamma_d, self.on_gamma_d))

    #1 2
    #0 3
    def get_corner_indices(self, initial_left_bottom, initial_right_top):
        corner_indices = np.zeros([4], dtype=np.long)
        corner_indices[0] = self.get_initial_index(
            [initial_left_bottom[0], initial_left_bottom[1]]
        )
        corner_indices[1] = self.get_initial_index(
            [initial_left_bottom[0], initial_right_top[1]]
        )
        corner_indices[2] = self.get_initial_index(
            [initial_right_top[0], initial_right_top[1]]
        )
        corner_indices[3] = self.get_initial_index(
            [initial_right_top[0], initial_left_bottom[1]]
        )
        return corner_indices

    def get_labels(self, points_number):
        labels = np.array([i for i in range(points_number)], dtype="long")
        return labels

    def get_point_types(
        self,
        points_number,
        initial_points,
        corner_indices,
        initial_left_bottom,
        initial_right_top,
    ):
        point_types = np.zeros([points_number, 3])
        point_types[:, 1] = (
            (initial_points[:, 0] == initial_left_bottom[0])
            | (initial_points[:, 1] == initial_left_bottom[1])
            | (initial_points[:, 0] == initial_right_top[0])
            | (initial_points[:, 1] == initial_right_top[1])
        )

        point_types[corner_indices[0]] = 0
        point_types[corner_indices[0], 2] = 1

        point_types[corner_indices[1]] = 0
        point_types[corner_indices[1], 2] = 1

        point_types[corner_indices[2]] = 0
        point_types[corner_indices[2], 2] = 1

        point_types[corner_indices[3]] = 0
        point_types[corner_indices[3], 2] = 1

        point_types[:, 0] = np.sum(point_types[:], axis=1) < 1

        return point_types

    def set_a_old(self, a):
        self.a_old = a

    def set_v_old(self, v):
        self.v_old = v

    def set_u_old(self, u):
        self.u_old = u

    def get_initial_index(self, point):
        return helpers.get_point_index(np.array(point), self.initial_points)

    @property
    def normalized_a_old(self):
        return self.rotate_to_upward(self.a_old)

    @property
    def normalized_points(self):
        return self.rotate_to_upward(self.moved_points - np.mean(self.moved_points, axis=0))

    @property
    def normalized_v_old(self):
        return self.rotate_to_upward(self.v_old - np.mean(self.v_old, axis=0))

    @property
    def normalized_u_old(self):
        #TODO: Rozkminić
        #normalized_u_old2 = self.rotate_to_upward(
        #    self.u_old - np.mean(self.u_old, axis=0)
        #) 
        return self.normalized_points - self.normalized_initial_points
        #return self.rotate_to_upward(self.moved_points - np.mean(self.moved_points, axis=0)) - self.normalized_initial_points

    @property
    def origin_u_old(self):
        return self.rotate_from_upward(self.normalized_u_old)

    @property
    def moved_points(self):
        return self.initial_points + self.u_old
    

    @property
    def up_right_vectors(self):
        v_left = np.subtract(self.corners_points[1], self.corners_points[0])
        v_left = v_left / np.linalg.norm(v_left)

        v_right = np.subtract(self.corners_points[2], self.corners_points[3])
        v_right = v_right / np.linalg.norm(v_right)

        v_top = np.subtract(self.corners_points[2], self.corners_points[1])
        v_top = v_top / np.linalg.norm(v_top)

        v_bottom = np.subtract(self.corners_points[3], self.corners_points[0])
        v_bottom = v_bottom / np.linalg.norm(v_bottom)

        v_up_mean = np.mean([v_left, v_right], axis=0)
        v_right_mean = np.mean([v_top, v_bottom], axis=0)

        return v_up_mean, v_right_mean

    @property
    def angle(self):
        up_vector, right_vector = self.up_right_vectors
        ax_vector = np.array([0, 1])
        angle = (2 * (up_vector[0] >= 0) - 1) * np.arccos(np.dot(up_vector, ax_vector))
        return angle

    def rotate(self, vectors, angle):
        s = np.sin(angle)
        c = np.cos(angle)

        rotated_vectors = np.zeros_like(vectors)
        rotated_vectors[:, 0] = vectors[:, 0] * c - vectors[:, 1] * s
        rotated_vectors[:, 1] = vectors[:, 0] * s + vectors[:, 1] * c

        return rotated_vectors

    def rotate_to_upward(self, vectors):
        return self.rotate(vectors, self.angle)

    def rotate_from_upward(self, vectors):
        return self.rotate(vectors, -self.angle)

    @property
    def corners_points(self):
        return self.moved_points[self.corner_indices]

    @property
    def edges_moved_points(self):
        return self.moved_points[self.edges]

    @property
    def edges_normalized_points(self):
        return self.normalized_points[self.edges]

    @property
    def cells_normalized_points(self):
        return self.normalized_points[self.cells]

    @property
    def points_number(self):
        return len(self.initial_points)

    @property
    def edges_number(self):
        return len(self.edges)



    def safe_divide(self, a, b):
        return np.divide(a, b, out=np.zeros_like(a), where=b!=0)

    def project(self, x, y):
        y_norm = np.linalg.norm(y, keepdims=True, axis=1)
        return self.safe_divide(np.sum(x*y, keepdims=True, axis=1), y_norm) \
            * self.safe_divide(y, y_norm)
    
    @property
    def normalized_v_nt_old(self):
        normalized_v_n_old = self.project(self.normalized_v_old, self.normalized_initial_points)
        normalized_v_t_old = self.normalized_v_old - normalized_v_n_old
        return normalized_v_n_old, normalized_v_t_old
