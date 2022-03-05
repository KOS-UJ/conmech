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


######################################################

# one time edge - boundary edge - not true in 3D
# boundary edges at the beginning
@njit
def get_ordered_edges_list(edges_matrix):
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
    boundary_edges_count = e
    for i in range(nodes_count):
        for j in range(nodes_count):
            if i != j:
                if edges_matrix[i, j] > 1:
                    edges[e] = np.array([i, j])
                    e += 1
    return edges, boundary_edges_count


@njit
def move_boundary_nodes_to_start_numba(
    unordered_points, unordered_cells, old_boundary_indices
):
    nodes_count = len(unordered_points)
    boundary_nodes_count = len(old_boundary_indices)

    points = np.zeros((nodes_count, unordered_points.shape[1]))
    cells = -(unordered_cells.copy() + 1)

    boundary_index = 0
    inner_index = nodes_count - 1
    for old_index in range(nodes_count):
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


######################################################

# @njit
def list_all_faces(elements):
    elements.sort(axis=1)
    elements_count, element_size = elements.shape
    dim = element_size - 1
    faces = np.zeros((element_size * elements_count, dim), dtype=np.int64)
    opposing_indices = np.zeros((element_size * elements_count), dtype=np.int64)
    i = 0
    for j in range(element_size):
        faces[i : i + elements_count, :j] = elements[:, :j]  # ignoring j-th column
        faces[i : i + elements_count, j:dim] = elements[:, j + 1 : element_size]
        opposing_indices[i : i + elements_count] = elements[:, j]
        i += elements_count
    return faces, opposing_indices


def extract_unique_elements(elements, opposing_indices):
    _, indices, count = np.unique(
        elements, axis=0, return_index=True, return_counts=True
    )
    unique_indices = indices[count == 1]
    return elements[unique_indices], opposing_indices[unique_indices]


def get_boundary_faces(elements):
    faces, opposing_indices = list_all_faces(elements)
    boundary_faces, boundary_internal_indices = extract_unique_elements(
        faces, opposing_indices
    )
    boundary_nodes_indices = np.unique(boundary_faces.flatten(), axis=0)
    return boundary_faces, boundary_nodes_indices, boundary_internal_indices


######################################################


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
def get_boundary_faces_internal_nodes(boundary_edges_count, edges, cells):
    boundary_faces_internal_nodes = np.zeros(boundary_edges_count, dtype=numba.int64)
    for i in range(boundary_edges_count):
        edge = edges[i]
        for cell in cells:
            internal_node = [node for node in cell if node not in edge]
            if len(internal_node) == 1:
                boundary_faces_internal_nodes[i] = internal_node[0]
                break
    return boundary_faces_internal_nodes


################


class SettingMesh:
    def __init__(
        self,
        mesh_type,
        mesh_density_x,
        mesh_density_y=None,
        scale_x=None,
        scale_y=None,
        is_adaptive=None,
        create_in_subprocess=None,
        is_3d=False,
    ):
        self.mesh_density_x = mesh_density_x
        self.mesh_density_y = mesh_density_y
        self.mesh_type = mesh_type
        self.scale_x = scale_x
        self.scale_y = scale_y
        self.is_adaptive = is_adaptive
        self.create_in_subprocess = create_in_subprocess
        self.is_3d = is_3d

        self.set_mesh()

    def remesh(self):
        self.set_mesh()

    def set_mesh(self):
        if self.is_3d:
            unordered_nodes, unordered_elements = mesh_builders.build_mesh(
                mesh_type=self.mesh_type, mesh_density_x=self.mesh_density_x
            )
        else:
            unordered_nodes, unordered_elements = mesh_builders.build_mesh(
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
        ) = self.clean_mesh(unordered_nodes, unordered_elements)

        (
            self.boundary_faces,
            self.boundary_nodes_indices,
            self.boundary_internal_indices,
        ) = get_boundary_faces(self.cells)

        self.base_seed_indices, self.closest_seed_index = get_base_seed_indices(self.initial_nodes)

        self.edges_matrix = get_edges_matrix(self.nodes_count, self.cells)
        self.edges, _ = get_ordered_edges_list(self.edges_matrix)
        self.boundary_faces_internal_nodes = get_boundary_faces_internal_nodes(
            self.boundary_edges_count, self.edges, self.cells
        )

        self.u_old = np.zeros_like(self.initial_nodes)
        self.v_old = np.zeros_like(self.initial_nodes)
        self.a_old = np.zeros_like(self.initial_nodes)

    def clean_mesh(self, unordered_nodes, unordered_elements):
        (
            old_boundary_faces,
            old_boundary_nodes_indices,
            old_boundary_internal_indices,
        ) = get_boundary_faces(unordered_elements)

        edges_matrix = get_edges_matrix(len(unordered_nodes), unordered_elements)
        old_ordered_edges, boundary_edges_count = get_ordered_edges_list(edges_matrix)

        old_boundary_edges = old_ordered_edges[:boundary_edges_count, :]
        old_boundary_indices = nph.get_occurances(old_boundary_edges)

        nodes, elements, boundary_nodes_count = move_boundary_nodes_to_start_numba(
            unordered_nodes, unordered_elements, old_boundary_indices,
        )
        return nodes, elements, boundary_nodes_count, boundary_edges_count

    def set_a_old(self, a):
        self.a_old = a

    def set_v_old(self, v):
        self.v_old = v

    def set_u_old(self, u):
        self.u_old = u

    @property
    def boundary_faces_normals(self):
        faces_nodes = self.moved_points[self.boundary_faces]
        internal_nodes = self.moved_points[self.boundary_internal_indices]
        tail_nodes, head_nodes = faces_nodes[:, 0], faces_nodes[:, 1]

        unoriented_normals = nph.get_tangential_2d(
            nph.normalize_euclidean_numba(head_nodes - tail_nodes)
        )
        external_orientation = (-1) * np.sign(
            nph.elementwise_dot(
                internal_nodes - tail_nodes, unoriented_normals, keepdims=True
            )
        )
        return unoriented_normals * external_orientation

    @property
    def normalized_boundary_faces_normals(self):
        return self.normalize_rotate(self.boundary_faces_normals)

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
        return self.normalized_points - self.normalized_initial_nodes

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
    def inner_nodes_count(self):
        return self.nodes_count - self.boundary_nodes_count

    @property
    def boundary_centers(self):
        return np.mean(self.moved_points[self.boundary_faces], axis=1)

    @property
    def normalized_boundary_centers(self):
        return np.mean(self.normalized_points[self.boundary_faces], axis=1)

    @property
    def edges_number(self):
        return len(self.edges)
