import deep_conmech.common.config as config
import deep_conmech.simulator.mesh.mesh_builders as mesh_builders
import numba
import numpy as np
from conmech.helpers import nph
from numba import njit

# import os, sys
# sys.path.append(os.path.abspath('../'))


@njit  # (parallel=True)
def get_edges_matrix(nodes_count, cells):
    edges_matrix = np.zeros((nodes_count, nodes_count), dtype=numba.int32)
    cell_vertices_number = len(cells[0])
    for cell in cells:  # TODO: prange?
        for i in range(cell_vertices_number):
            for j in range(cell_vertices_number):
                if i != j:
                    edges_matrix[cell[i], cell[j]] += 1.0
    return edges_matrix


# one time edge - boundary edge
# bondary edges at the beginning
@njit
def get_edges_list(edges_matrix):
    edges_number = np.sum(edges_matrix > 0, dtype=numba.int64)
    nodes_count = len(edges_matrix[0])
    edges = np.zeros((edges_number, 2), dtype=numba.int64)
    e = 0
    for i in range(nodes_count):
        for j in range(nodes_count):
            if i != j:
                if edges_matrix[i, j] == 1:
                    edges[e] = np.array([i, j])
                    e += 1
    for i in range(nodes_count):
        for j in range(nodes_count):
            if i != j:
                if edges_matrix[i, j] > 1:
                    edges[e] = np.array([i, j])
                    e += 1
    return edges


##############


@njit
def move_boundary_points_to_start(
    unordered_points, unordered_cells, old_attached_point_indices, old_boundary_indices
):
    nodes_count = len(old_attached_point_indices)
    boundary_nodes_count = len(old_boundary_indices)

    points = np.zeros((nodes_count, unordered_points.shape[1]))
    cells = -(unordered_cells.copy() + 1)

    boundary_index = 0
    inner_index = nodes_count - 1
    for old_index in old_attached_point_indices:
        point = unordered_points[old_index]

        if old_index in old_boundary_indices:
            new_index = boundary_index
            boundary_index += 1
        else:
            new_index = inner_index
            inner_index -= 1

        points[new_index] = point
        cells = np.where(cells == -(old_index + 1), new_index, cells)

    return points, cells, boundary_nodes_count


############


@njit
def get_furthest_apart_numba(points, variable):
    max_dist = 0.0
    max_i, max_j = 0, 0
    nodes_count = len(points)
    for i in range(nodes_count):
        for j in range(i, nodes_count):
            dist = np.abs(points[i, variable] - points[j, variable])
            if dist > max_dist:
                max_dist = dist
                max_i, max_j = i, j

    if points[max_i, variable] < points[max_j, variable]:
        return [max_i, max_j]
    else:
        return [max_j, max_i]


############


@njit
def get_boundary_edges_internal_nodes(boundary_edges_count, edges, cells):
    boundary_edges_internal_nodes = np.zeros(boundary_edges_count, dtype=numba.int64)
    for i in range(boundary_edges_count):
        edge = edges[i]
        for cell in cells:
            internal_node = [node for node in cell if node not in edge]
            if len(internal_node) == 1:
                boundary_edges_internal_nodes[i] = internal_node[0]
                break
    return boundary_edges_internal_nodes


################


class SettingMesh:
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
        self.mesh_density_x = mesh_density_x
        self.mesh_density_y = mesh_density_y
        self.mesh_type = mesh_type
        self.scale_x = scale_x
        self.scale_y = scale_y
        self.is_adaptive = is_adaptive
        self.create_in_subprocess = create_in_subprocess

        self.set_mesh()

    def remesh(self):
        self.set_mesh()

    def set_mesh(self):
        unordered_points, unordered_cells = mesh_builders.build_mesh(
            mesh_type=self.mesh_type,
            mesh_density_x=self.mesh_density_x,
            mesh_density_y=self.mesh_density_y,
            scale_x=self.scale_x,
            scale_y=self.scale_y,
            is_adaptive=self.is_adaptive,
            create_in_subprocess=self.create_in_subprocess,
        )
        (
            self.initial_points,
            self.cells,
            self.boundary_nodes_count,
            self.boundary_edges_count,
        ) = self.clean_mesh(unordered_points, unordered_cells)

        self.rotation_reference_indices = get_furthest_apart_numba(
            self.boundary_points_initial, 1
        )

        self.edges_matrix = get_edges_matrix(self.nodes_count, self.cells)
        self.edges = get_edges_list(self.edges_matrix)
        self.boundary_edges_internal_nodes = get_boundary_edges_internal_nodes(
            self.boundary_edges_count, self.edges, self.cells
        )

        self.set_u_old(np.zeros_like(self.initial_points))
        self.set_v_old(np.zeros_like(self.initial_points))
        self.set_a_old(np.zeros_like(self.initial_points))

    def clean_mesh(self, unordered_points, unordered_cells):
        nodes_count = len(unordered_points)
        edges_matrix = get_edges_matrix(nodes_count, unordered_cells)
        boundary_edges_count = np.sum(edges_matrix == 1, dtype=np.int64)
        edges = get_edges_list(edges_matrix)
        boundary_edges = edges[:boundary_edges_count, :]

        old_attached_point_indices = nph.get_occurances(unordered_cells)
        old_boundary_indices = nph.get_occurances(boundary_edges)

        points, cells, boundary_nodes_count = move_boundary_points_to_start(
            unordered_points,
            unordered_cells,
            old_attached_point_indices,
            old_boundary_indices,
        )
        return points, cells, boundary_nodes_count, boundary_edges_count

    def set_a_old(self, a):
        self.a_old = a

    def set_v_old(self, v):
        self.v_old = v

    def set_u_old(self, u):
        self.u_old = u

    def get_initial_index(self, point):
        return nph.get_point_index_numba(np.array(point), self.initial_points)

    @property
    def boundary_edges_normals(self):
        edges_nodes = self.moved_points[self.boundary_edges]
        internal_nodes = self.moved_points[self.boundary_edges_internal_nodes]
        tail_nodes, head_nodes = edges_nodes[:, 0], edges_nodes[:, 1]

        unoriented_normals = nph.get_tangential_2d(
            nph.normalize_euclidean_numba(head_nodes - tail_nodes)
        )
        internal_orientation = np.sign(
            nph.elementwise_dot(internal_nodes - tail_nodes, unoriented_normals)
        )
        result = unoriented_normals * (-1) * internal_orientation.reshape(-1, 1)
        return result

    @property
    def normalized_boundary_edges_normals(self):
        return self.rotate_to_upward(self.boundary_edges_normals)

    @property
    def normalized_a_old(self):
        return self.rotate_to_upward(self.a_old)

    @property
    def mean_moved_points(self):
        return np.mean(self.moved_points, axis=0)

    @property
    def mean_initial_points(self):
        return np.mean(self.initial_points, axis=0)

    @property
    def normalized_points(self):
        return self.rotate_to_upward(self.moved_points - self.mean_moved_points)

    @property
    def normalized_initial_points(self):
        return self.initial_points - self.mean_initial_points

    @property
    def normalized_v_old(self):
        return self.rotate_to_upward(self.v_old - np.mean(self.v_old, axis=0))

    @property
    def normalized_u_old(self):
        # TODO: Rozkminić
        # normalized_u_old2 = self.rotate_to_upward(
        #    self.u_old - np.mean(self.u_old, axis=0)
        # )
        return self.normalized_points - self.normalized_initial_points
        # return self.rotate_to_upward(self.moved_points - np.mean(self.moved_points, axis=0)) - self.normalized_initial_points

    @property
    def origin_u_old(self):
        return self.rotate_from_upward(self.normalized_u_old)

    @property
    def moved_points(self):
        return self.initial_points + self.u_old

    @property
    def initial_reference_points(self):
        return self.initial_points[self.rotation_reference_indices]

    @property
    def moved_reference_points(self):
        return self.moved_points[self.rotation_reference_indices]

    @property
    def normalized_reference_points(self):
        return self.normalized_points[self.rotation_reference_indices]

    @property
    def initial_base_seed(self):
        return np.array([self.initial_reference_points[1] - self.initial_reference_points[0]])

    @property
    def moved_base_seed(self):
        return np.array([self.moved_reference_points[1] - self.moved_reference_points[0]])

    @property
    def to_rotated_base_seed(self):
        return nph.get_in_base(self.moved_base_seed, self.initial_base_seed) if config.NORMALIZE_ROTATE else np.array((0.0, 1.0))
    @property
    def from_rotated_base_seed(self):
        return nph.get_in_base(self.initial_base_seed, self.moved_base_seed) if config.NORMALIZE_ROTATE else np.array((0.0, 1.0))

    def rotate_to_upward(self, vectors):
        return nph.get_in_base(vectors, self.to_rotated_base_seed)

    def rotate_from_upward(self, vectors):
        return nph.get_in_base(vectors, self.from_rotated_base_seed)

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
    def nodes_count(self):
        return len(self.initial_points)

    @property
    def boundary_edges(self):
        return self.edges[: self.boundary_edges_count, :]

    @property
    def boundary_points_initial(self):
        return self.initial_points[: self.boundary_edges_count, :]

    @property
    def inner_nodes_count(self):
        return self.nodes_count - self.boundary_nodes_count

    @property
    def boundary_centers(self):
        return np.mean(self.moved_points[self.boundary_edges], axis=1)

    @property
    def normalized_boundary_centers(self):
        return np.mean(self.normalized_points[self.boundary_edges], axis=1)

    @property
    def edges_number(self):
        return len(self.edges)

    def safe_divide(self, a, b):
        return np.divide(a, b, out=np.zeros_like(a), where=b != 0)

    def project(self, x, y):
        y_norm = np.linalg.norm(y, keepdims=True, axis=1)
        return self.safe_divide(
            np.sum(x * y, keepdims=True, axis=1), y_norm
        ) * self.safe_divide(y, y_norm)

    @property
    def normalized_v_nt_old(self):
        normalized_v_n_old = self.project(
            self.normalized_v_old, self.normalized_initial_points
        )
        normalized_v_t_old = self.normalized_v_old - normalized_v_n_old
        return normalized_v_n_old, normalized_v_t_old
