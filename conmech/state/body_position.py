import copy
from ctypes import ArgumentError
from typing import Callable, Optional

import jax
import jax.numpy as jnp
import numpy as np

from conmech.helpers import jxh, lnh, nph
from conmech.mesh.boundaries_description import BoundariesDescription
from conmech.mesh.mesh import Mesh
from conmech.properties.mesh_properties import MeshProperties
from conmech.properties.schedule import Schedule


def _get_unoriented_normals_2d_jax(faces_nodes):
    tail_nodes, head_nodes = faces_nodes[:, 0], faces_nodes[:, 1]

    unoriented_normals = jxh.get_tangential_2d(jxh.normalize_euclidean(head_nodes - tail_nodes))
    return tail_nodes, unoriented_normals


def _get_unoriented_normals_3d_jax(faces_nodes):
    tail_nodes, head_nodes1, head_nodes2 = [faces_nodes[:, i, :] for i in range(3)]

    unoriented_normals = jxh.normalize_euclidean(
        jnp.cross(head_nodes1 - tail_nodes, head_nodes2 - tail_nodes)
    )
    return tail_nodes, unoriented_normals


def _get_boundary_surfaces_normals_jax(
    moved_nodes, boundary_surfaces, boundary_internal_indices, dimension
):
    faces_nodes = moved_nodes[boundary_surfaces]

    if dimension == 2:
        tail_nodes, unoriented_normals = _get_unoriented_normals_2d_jax(faces_nodes)
    elif dimension == 3:
        tail_nodes, unoriented_normals = _get_unoriented_normals_3d_jax(faces_nodes)

    internal_nodes = moved_nodes[boundary_internal_indices]
    external_orientation = (-1) * jnp.sign(
        nph.elementwise_dot(internal_nodes - tail_nodes, unoriented_normals, keepdims=True)
    )
    return unoriented_normals * external_orientation


def _aggrergate_boundary_surfaces_jax(data, boundary_surfaces, considered_nodes_count, agg_fun):
    return agg_fun(
        jnp.array(
            [
                jax.ops.segment_sum(
                    data=data,
                    segment_ids=boundary_surfaces[:, i],
                    num_segments=considered_nodes_count,
                )
                for i in range(boundary_surfaces.shape[1])
            ]
        ),
        axis=0,
    )


def _get_boundary_normals_jax(
    moved_nodes, boundary_surfaces, boundary_internal_indices, considered_nodes_count
):
    print("get_boundary_normals_jax")
    dimension = moved_nodes.shape[1]
    boundary_surfaces_normals = _get_boundary_surfaces_normals_jax(
        moved_nodes, boundary_surfaces, boundary_internal_indices, dimension
    )
    boundary_normals = _aggrergate_boundary_surfaces_jax(
        data=boundary_surfaces_normals,
        boundary_surfaces=boundary_surfaces,
        considered_nodes_count=considered_nodes_count,
        agg_fun=jnp.mean,
    )
    boundary_normals = jxh.normalize_euclidean(boundary_normals)
    return boundary_normals


def _get_element_volume_part_jax(moved_nodes, boundary_surfaces):
    print("get_element_volume_part_jax")
    moved_boundary_nodes = moved_nodes[boundary_surfaces]
    dimension = moved_nodes.shape[1]
    nodes_count = boundary_surfaces.shape[1]

    if dimension == 2:
        volume = jxh.euclidean_norm(moved_boundary_nodes[:, 1, :] - moved_boundary_nodes[:, 0, :])
    elif dimension == 3:
        volume = 0.5 * jxh.euclidean_norm(
            jnp.cross(
                moved_boundary_nodes[:, 1, :] - moved_boundary_nodes[:, 0, :],
                moved_boundary_nodes[:, 2, :] - moved_boundary_nodes[:, 0, :],
            ),
            keepdims=True,
        )
    else:
        raise ArgumentError
    return volume / nodes_count


def get_surface_per_boundary_node_jax(moved_nodes, boundary_surfaces, considered_nodes_count):
    print("get_surface_per_boundary_node_jax")
    element_volume_part = _get_element_volume_part_jax(moved_nodes, boundary_surfaces)

    surface_per_boundary_node = _aggrergate_boundary_surfaces_jax(
        data=element_volume_part,
        boundary_surfaces=boundary_surfaces,
        considered_nodes_count=considered_nodes_count,
        agg_fun=jnp.sum,
    )
    return nph.stack_column(surface_per_boundary_node)


def mesh_normalization_decorator(func: Callable):
    def inner(self, *args, **kwargs):
        saved_normalize = self.normalize
        self.normalize = True
        if hasattr(self, "reduced"):
            self.reduced.normalize = True
        returned_value = func(self, *args, **kwargs)
        self.normalize = saved_normalize
        if hasattr(self, "reduced"):
            self.reduced.normalize = saved_normalize
        return returned_value

    return inner


# pylint: disable=R0904
class BodyPosition:
    def __init__(
        self,
        mesh_prop: MeshProperties,
        schedule: Schedule,
        normalize: bool = False,
        boundaries_description: Optional[BoundariesDescription] = None,
        create_in_subprocess: bool = False,
    ):
        if boundaries_description is None:
            boundaries_description = BoundariesDescription(contact=None, dirichlet=None)

        self.mesh = Mesh(
            mesh_prop=mesh_prop,
            boundaries_description=boundaries_description,
            create_in_subprocess=create_in_subprocess,
        )
        self.normalize = normalize

        self.schedule = schedule
        self.__displacement_old = np.zeros_like(self.initial_nodes)
        self.__velocity_old = np.zeros_like(self.initial_nodes)
        self.exact_acceleration = np.zeros_like(self.initial_nodes)
        self.moved_base = None
        self.boundary_normals = np.zeros_like(self.boundary_nodes)
        self.set_boundary_normals_jax()

    def prepare(self, inner_forces):
        _ = inner_forces
        self.set_boundary_normals_jax()

    def _normalize_shift(self, vectors):
        _ = self
        if not self.normalize:
            return vectors
        return vectors - np.mean(vectors, axis=0)

    @property
    def boundaries(self):
        return self.mesh.boundaries

    @property
    def edges(self):
        return self.mesh.edges

    @property
    def elements(self):
        return self.mesh.elements

    @property
    def mesh_prop(self):
        return self.mesh.mesh_prop

    @property
    def initial_nodes(self):
        return self.mesh.initial_nodes

    @property
    def normalized_initial_nodes(self):
        return self._normalize_shift(self.initial_nodes)

    @property
    def input_initial_nodes(self):
        return self.normalized_initial_nodes

    @property
    def boundary_surfaces(self):
        return self.mesh.boundary_surfaces

    @property
    def contact_boundary(self):
        return self.mesh.contact_boundary

    @property
    def neumann_boundary(self):
        return self.mesh.neumann_boundary

    @property
    def dirichlet_boundary(self):
        return self.mesh.dirichlet_boundary

    @property
    def boundary_internal_indices(self):
        return self.mesh.boundary_internal_indices

    @property
    def boundary_nodes_count(self):
        return self.mesh.boundary_nodes_count

    @property
    def contact_nodes_count(self):
        return self.mesh.contact_nodes_count

    @property
    def dirichlet_nodes_count(self):
        return self.mesh.dirichlet_nodes_count

    @property
    def neumann_nodes_count(self):
        return self.mesh.neumann_nodes_count

    @property
    def independent_nodes_count(self):
        return self.mesh.independent_nodes_count

    @property
    def free_nodes_count(self):
        return self.mesh.free_nodes_count

    @property
    def boundary_indices(self):
        return self.mesh.boundary_indices

    @property
    def initial_boundary_nodes(self):
        return self.mesh.initial_boundary_nodes

    @property
    def contact_indices(self):
        return self.mesh.contact_indices

    @property
    def neumann_indices(self):
        return self.mesh.neumann_indices

    @property
    def dirichlet_indices(self):
        return self.mesh.dirichlet_indices

    @property
    def independent_indices(self):
        return self.mesh.independent_indices

    @property
    def free_indices(self):
        return self.mesh.free_indices

    @property
    def dimension(self):
        return self.mesh.dimension

    @property
    def nodes_count(self):
        return self.mesh.nodes_count

    @property
    def elements_count(self):
        return self.mesh.elements_count

    @property
    def boundary_surfaces_count(self):
        return self.mesh.boundary_surfaces_count

    @property
    def inner_nodes_count(self):
        return self.mesh.inner_nodes_count

    @property
    def edges_number(self):
        return self.mesh.edges_number

    @property
    def centered_initial_nodes(self):
        return self.mesh.centered_initial_nodes

    @property
    def displacement_old(self):
        return self.__displacement_old

    @property
    def velocity_old(self):
        return self.__velocity_old

    def set_displacement_old(self, displacement):
        self.__displacement_old = displacement

    def set_velocity_old(self, velocity):
        self.__velocity_old = velocity

    @property
    def position(self):
        return np.mean(self.displacement_old, axis=0)

    @property
    def time_step(self):
        return self.schedule.time_step

    def get_copy(self):
        return copy.deepcopy(self)

    def iterate_self(self, acceleration, temperature=None):
        # Test:
        # x = self.from_normalized_displacement(
        #     self.to_normalized_displacement(acceleration)
        # )
        # assert np.allclose(x, acceleration)
        # print(np.linalg.norm(acceleration), np.linalg.norm(x- acceleration))

        _ = temperature
        velocity = self.velocity_old + self.time_step * acceleration
        displacement = self.displacement_old + self.time_step * velocity

        self.set_displacement_old(displacement)
        self.set_velocity_old(velocity)

        return self

    def normalize_rotate(self, vectors):
        if not self.normalize:
            return vectors
        return lnh.get_in_base(vectors, self.moved_base)

    def denormalize_rotate(self, vectors):
        if not self.normalize:
            return vectors
        return lnh.get_in_base(vectors, np.linalg.inv(self.moved_base))

    def normalize_shift_and_rotate(self, vectors):
        return self.normalize_rotate(self._normalize_shift(vectors))

    @property
    def moved_nodes(self):
        return self.initial_nodes + self.displacement_old

    @property
    def normalized_nodes(self):
        return self.normalize_shift_and_rotate(self.moved_nodes)

    @property
    def boundary_nodes(self):
        return self.moved_nodes[self.boundary_indices]

    @property
    def normalized_boundary_nodes(self):
        return self.normalized_nodes[self.boundary_indices]

    def get_normalized_boundary_normals_jax(self):
        return self.normalize_rotate(self.boundary_normals)

    @property
    def mean_moved_nodes(self):
        return np.mean(self.moved_nodes, axis=0)

    @property
    def edges_moved_nodes(self):
        return self.moved_nodes[self.edges]

    @property
    def edges_normalized_nodes(self):
        return self.normalized_nodes[self.edges]

    @property
    def elements_normalized_nodes(self):
        return self.normalized_nodes[self.elements]

    @property
    def boundary_centers(self):
        return np.mean(self.moved_nodes[self.boundary_surfaces], axis=1)

    @property
    def normalized_velocity_old(self):
        return self.normalize_rotate(self.velocity_old)  # normalize_shift_and_rotate

    @property
    def normalized_displacement_old(self):
        return self.normalized_nodes - self.normalized_initial_nodes

    def set_boundary_normals_jax(self):
        self.boundary_normals[:] = jax.jit(
            _get_boundary_normals_jax, static_argnames=["considered_nodes_count"]
        )(
            moved_nodes=self.moved_nodes,
            boundary_surfaces=self.boundary_surfaces,
            boundary_internal_indices=self.boundary_internal_indices,
            considered_nodes_count=self.boundary_nodes_count,
        )

    def get_surface_per_boundary_node_jax(self):
        return jax.jit(
            get_surface_per_boundary_node_jax, static_argnames=["considered_nodes_count"]
        )(
            moved_nodes=self.moved_nodes,
            boundary_surfaces=self.boundary_surfaces,
            considered_nodes_count=self.boundary_nodes_count,
        )

    @property
    def input_velocity_old(self):
        return self.normalized_velocity_old

    @property
    def input_displacement_old(self):
        return self.normalized_displacement_old

    @property
    def centered_nodes(self):
        return lnh.get_in_base(
            (self.moved_nodes - np.mean(self.moved_nodes, axis=0)), self.moved_base
        )
