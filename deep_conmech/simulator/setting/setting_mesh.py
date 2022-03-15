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
def get_closest_to_axis(nodes, variable):
    min_error = 1.0
    final_i, final_j = 0, 0
    nodes_count = len(nodes)
    for i in range(nodes_count):
        for j in range(i + 1, nodes_count):
            # dist = np.abs(nodes[i, variable] - nodes[j, variable])
            error = nph.euclidean_norm_numba(
                np.delete(nodes[i], variable) - np.delete(nodes[j], variable)
            )
            if error < min_error:
                min_error, final_i, final_j = error, i, j

    correct_order = nodes[final_i, variable] < nodes[final_j, variable]
    indices = (final_i, final_j) if correct_order else (final_j, final_i)
    return np.array([error, indices[0], indices[1]])


def get_base_seed_indices(nodes):
    dim = nodes.shape[1]
    base_seed_indices = np.zeros((dim, 2), dtype=np.int64)
    errors = np.zeros(dim)
    for i in range(dim):
        result = get_closest_to_axis(nodes, i)
        errors[i] = result[0]
        base_seed_indices[i] = result[1:].astype(np.int64)
        # print(f"MIN ERROR for variable {i}: {errors[i]}")
    return base_seed_indices, np.argmin(errors)


def get_base(nodes, base_seed_indices, closest_seed_index):
    base_seed_initial_nodes = nodes[base_seed_indices]
    base_seed = base_seed_initial_nodes[..., 1, :] - base_seed_initial_nodes[..., 0, :]
    return nph.complete_base(base_seed, closest_seed_index)


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
            self.initial_nodes,
            self.cells,
            self.boundary_nodes_count,
            self.boundary_edges_count,
        ) = self.clean_mesh(unordered_points, unordered_cells)

        self.base_seed_indices, self.closest_seed_index = get_base_seed_indices(
            self.boundary_points_initial
        )

        self.edges_matrix = get_edges_matrix(self.nodes_count, self.cells)
        self.edges = get_edges_list(self.edges_matrix)
        self.boundary_edges_internal_nodes = get_boundary_edges_internal_nodes(
            self.boundary_edges_count, self.edges, self.cells
        )

        self.set_u_old(np.zeros_like(self.initial_nodes))
        self.set_v_old(np.zeros_like(self.initial_nodes))
        self.set_a_old(np.zeros_like(self.initial_nodes))

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
        return nph.get_point_index_numba(np.array(point), self.initial_nodes)

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
        return self.normalize_rotate(self.boundary_edges_normals)

    @property
    def normalized_a_old(self):
        return self.normalize_rotate(self.a_old)

    @property
    def mean_moved_points(self):
        return np.mean(self.moved_points, axis=0)

    @property
    def mean_initial_nodes(self):
        return np.mean(self.initial_nodes, axis=0)

    @property
    def normalized_points(self):
        return self.normalize_rotate(self.moved_points - self.mean_moved_points)

    @property
    def normalized_initial_nodes(self):
        return self.initial_nodes - self.mean_initial_nodes

    @property
    def normalized_v_old(self):
        return self.normalize_rotate(self.v_old - np.mean(self.v_old, axis=0))

    @property
    def normalized_u_old(self):
        # TODO: Rozkminić
        # normalized_u_old2 = self.normalize_rotate(
        #    self.u_old - np.mean(self.u_old, axis=0)
        # )
        return self.normalized_points - self.normalized_initial_nodes
        # return self.normalize_rotate(self.moved_points - np.mean(self.moved_points, axis=0)) - self.normalized_initial_nodes

    @property
    def origin_u_old(self):
        return self.denormalize_rotate(self.normalized_u_old)

    @property
    def moved_points(self):
        return self.initial_nodes + self.u_old

    @property
    def moved_base(self):
        return get_base(
            self.moved_points, self.base_seed_indices, self.closest_seed_index
        )

    def normalize_rotate(self, vectors):
        return nph.get_in_base(vectors, self.moved_base)

    def denormalize_rotate(self, vectors):
        return nph.get_in_base(vectors, np.linalg.inv(self.moved_base))

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
        return len(self.initial_nodes)

    @property
    def boundary_edges(self):
        return self.edges[: self.boundary_edges_count, :]

    @property
    def boundary_points_initial(self):
        return self.initial_nodes[: self.boundary_edges_count, :]

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
            self.normalized_v_old, self.normalized_initial_nodes
        )
        normalized_v_t_old = self.normalized_v_old - normalized_v_n_old
        return normalized_v_n_old, normalized_v_t_old
