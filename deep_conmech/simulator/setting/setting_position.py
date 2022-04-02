from ctypes import ArgumentError
from dataclasses import dataclass
from typing import Callable

import numpy as np
from conmech.dataclass.body_properties import StaticBodyProperties
from conmech.dataclass.mesh_data import MeshData
from conmech.dataclass.schedule import Schedule
from conmech.helpers import nph
from conmech.solvers.solver_methods import njit
from deep_conmech.simulator.dynamics.dynamics import Dynamics



def get_base(nodes, base_seed_indices, closest_seed_index):
    base_seed_initial_nodes = nodes[base_seed_indices]
    base_seed = base_seed_initial_nodes[..., 1, :] - base_seed_initial_nodes[..., 0, :]
    return nph.complete_base(base_seed, closest_seed_index)



    
def get_unoriented_normals_2d(faces_nodes):
    tail_nodes, head_nodes = faces_nodes[:, 0], faces_nodes[:, 1]

    unoriented_normals = nph.get_tangential_2d(
        nph.normalize_euclidean_numba(head_nodes - tail_nodes)
    )
    return tail_nodes, unoriented_normals


def get_unoriented_normals_3d(faces_nodes):
    tail_nodes, head_nodes1, head_nodes2 = [faces_nodes[:, i, :] for i in range(3)]

    unoriented_normals = nph.normalize_euclidean_numba(
        np.cross(head_nodes1 - tail_nodes, head_nodes2 - tail_nodes)
    )
    return tail_nodes, unoriented_normals


def get_boundary_faces_normals(moved_nodes, boundary_faces, boundary_internal_indices):
    dim = moved_nodes.shape[1]
    faces_nodes = moved_nodes[boundary_faces]

    if dim == 2:
        tail_nodes, unoriented_normals = get_unoriented_normals_2d(faces_nodes)
    elif dim == 3:
        tail_nodes, unoriented_normals = get_unoriented_normals_3d(faces_nodes)
    else:
        raise ArgumentError

    internal_nodes = moved_nodes[boundary_internal_indices]
    external_orientation = (-1) * np.sign(
        nph.elementwise_dot(
            internal_nodes - tail_nodes, unoriented_normals, keepdims=True
        )
    )
    return unoriented_normals * external_orientation




@njit
def get_boundary_nodes_data_numba(
        boundary_faces_normals, boundary_faces, boundary_nodes_count, moved_nodes
):
    dim = boundary_faces_normals.shape[1]
    boundary_normals = np.zeros((boundary_nodes_count, dim), dtype=np.float64)
    boundary_nodes_volume = np.zeros((boundary_nodes_count, 1), dtype=np.float64)

    for i in range(boundary_nodes_count):
        node_faces_count = 0
        for j in range(len(boundary_faces)):
            if np.any(boundary_faces[j] == i):
                node_faces_count += 1
                boundary_normals[i] += boundary_faces_normals[j]

                face_nodes = moved_nodes[boundary_faces[j]]
                boundary_nodes_volume[i] += element_volume_part_numba(face_nodes)

        boundary_normals[i] /= node_faces_count

    boundary_normals = nph.normalize_euclidean_numba(boundary_normals)
    return boundary_normals, boundary_nodes_volume

@njit
def element_volume_part_numba(face_nodes):
    dim = face_nodes.shape[1]
    nodes_count = face_nodes.shape[0]
    if dim == 2:
        volume = nph.euclidean_norm_numba(face_nodes[0] - face_nodes[1])
    elif dim == 3:
        volume = 0.5 * nph.euclidean_norm_numba(
            np.cross(face_nodes[1] - face_nodes[0], face_nodes[2] - face_nodes[0])
        )
    else:
        raise ArgumentError
    return volume / nodes_count




class SettingPosition(Dynamics):
    def __init__(
            self,
            mesh_data: MeshData,
            body_prop: StaticBodyProperties,
            schedule: Schedule,
            normalize_by_rotation: bool,
            is_dirichlet: Callable = (lambda _: False),
            is_contact: Callable = (lambda _: True),
            with_schur_complement_matrices: bool = True,
            create_in_subprocess: bool = False,
    ):
        super().__init__(
            mesh_data=mesh_data,
            body_prop=body_prop,
            schedule=schedule,
            normalize_by_rotation=normalize_by_rotation,
            is_dirichlet=is_dirichlet,
            is_contact=is_contact,
            with_schur_complement_matrices=with_schur_complement_matrices,
            create_in_subprocess=create_in_subprocess,
        )

        self.state_dict["u_old"] = np.zeros_like(self.initial_nodes)
        self.state_dict["v_old"] = np.zeros_like(self.initial_nodes)
        self.state_dict["a_old"] = np.zeros_like(self.initial_nodes)
        self.clear()


    def remesh(self):
        super().remesh()
        self.clear()


    @property
    def u_old(self):
        return self.state_dict["u_old"]

    @property
    def v_old(self):
        return self.state_dict["v_old"]
        
    @property
    def a_old(self):
        return self.state_dict["a_old"]



    def set_a_old(self, a):
        self.state_dict["a_old"] = a

    def set_v_old(self, v):
        self.state_dict["v_old"] = v

    def set_u_old(self, u):
        self.state_dict["u_old"] = u




    @property
    def moved_base(self):
        return get_base(
            self.moved_nodes, self.base_seed_indices, self.closest_seed_index
        )
        
    def normalize_rotate(self, vectors):
        return (
            nph.get_in_base(vectors, self.moved_base)
            if self.normalize_by_rotation
            else vectors
        )

    def denormalize_rotate(self, vectors):
        return nph.get_in_base(vectors, np.linalg.inv(self.moved_base))



    @property
    def moved_nodes(self):
        return self.initial_nodes + self.u_old

    @property
    def normalized_points(self):
        return self.normalize_rotate(self.moved_nodes - self.mean_moved_nodes)

    @property
    def boundary_nodes(self):
        return self.moved_nodes[self.boundary_indices]

    @property
    def normalized_boundary_nodes(self):
        return self.normalized_points[self.boundary_indices]

    @property
    def normalized_boundary_normals(self):
        return self.normalize_rotate(self.boundary_normals)

    @property
    def normalized_a_old(self):
        return self.normalize_rotate(self.a_old)

    @property
    def mean_moved_nodes(self):
        return np.mean(self.moved_nodes, axis=0)


    @property
    def edges_moved_nodes(self):
        return self.moved_nodes[self.edges]

    @property
    def edges_normalized_points(self):
        return self.normalized_points[self.edges]

    @property
    def elements_normalized_points(self):
        return self.normalized_points[self.elements]

    @property
    def boundary_centers(self):
        return np.mean(self.moved_nodes[self.boundary_faces], axis=1)



    @property
    def rotated_v_old(self):
        return self.normalize_rotate(self.v_old)

    @property
    def normalized_v_old(self):
        return self.normalize_rotate(self.v_old - np.mean(self.v_old, axis=0))

    @property
    def normalized_u_old(self):
        return self.normalized_points - self.normalized_initial_nodes

    @property
    def origin_u_old(self):
        return self.denormalize_rotate(self.normalized_u_old)






    def clear(self):
        self.boundary_normals = None
        self.boundary_nodes_volume = None


    def prepare(self):
        boundary_faces_normals = get_boundary_faces_normals(
            self.moved_nodes, self.boundary_faces, self.boundary_internal_indices
        )
        (
            self.boundary_normals,
            self.boundary_nodes_volume,
        ) = get_boundary_nodes_data_numba(
            boundary_faces_normals,
            self.boundary_faces,
            self.boundary_nodes_count,
            self.moved_nodes,
        )
