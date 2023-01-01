from dataclasses import dataclass
from typing import NamedTuple, Optional

import jax.experimental.sparse
import jax.interpreters.xla
import jax.numpy as jnp
import jax.scipy
import numba
import numpy as np
from jax import lax

from conmech.dynamics.factory.dynamics_factory_method import ConstMatrices, get_dynamics
from conmech.helpers import cmh, jxh
from conmech.helpers.lnh import complete_base
from conmech.mesh.boundaries_description import BoundariesDescription
from conmech.properties.body_properties import (
    StaticBodyProperties,
    TemperatureBodyProperties,
)
from conmech.properties.mesh_properties import MeshProperties
from conmech.properties.schedule import Schedule
from conmech.solvers.optimization.schur_complement import SchurComplement
from conmech.state.body_position import BodyPosition


@numba.njit
def get_edges_features_list_numba(edges_number, edges_features_matrix):
    nodes_count = len(edges_features_matrix[0])
    edges_features = np.zeros((edges_number + nodes_count, 8))
    edge_id = 0
    for i in range(nodes_count):
        for j in range(nodes_count):
            if np.any(edges_features_matrix[i, j]):
                edges_features[edge_id] = edges_features_matrix[i, j]
                edge_id += 1
    return edges_features


# TODO: #75
@dataclass
class SolverMatrices:
    def __init__(self):
        self.lhs_sparse: jax.experimental.sparse.BCOO
        # TODO: #75 move to schur (careful - some properties are used by net)
        self.free_x_contact: np.ndarray
        self.contact_x_free: np.ndarray
        self.free_x_free_inverted: np.ndarray

        self.lhs_boundary: jax.interpreters.xla.DeviceArray

        self.lhs_sparse_jax: jax.experimental.sparse.BCOO
        self.lhs_acceleration_jax: np.ndarray
        self.lhs_preconditioner_jax: np.ndarray
        self.contact_x_contact: np.ndarray
        self.free_x_free: np.ndarray

        self.lhs_temperature_sparse_jax: jax.experimental.sparse.BCOO  # sparse.base.spmatrix
        self.temperature_boundary: np.ndarray
        self.temperature_free_x_contact: np.ndarray
        self.temperature_contact_x_free: np.ndarray
        self.temperature_free_x_free_inv: np.ndarray
        self.temperature_free_x_free: np.ndarray

    @property
    def lhs(self):
        return jxh.to_dense_np(self.lhs_sparse)

    @property
    def lhs_temperature(self):
        return jxh.to_dense_np(self.lhs_temperature_sparse_jax)


@dataclass
class DynamicsConfiguration:
    create_in_subprocess: bool = False
    with_lhs: bool = False
    with_schur: bool = False


def _get_jac(value, dx_big_jax):
    dimension = value.shape[1]
    result0 = (
        (dx_big_jax @ jnp.tile(value, (dimension, 1)))
        .reshape(dimension, -1, dimension)
        .swapaxes(0, 1)
        .transpose((0, 2, 1))
    )
    return result0


def _get_deform_grad(value, dx_big_jax):
    dimension = value.shape[1]
    identity = jnp.eye(dimension)
    return _get_jac(value, dx_big_jax) + identity


class _GetRotationState(NamedTuple):
    rotation: jnp.ndarray
    norm: float
    iteration: int
    success: bool


@jax.jit
def _get_rotation_jax(displacement, dx_big):
    max_iter = 30
    max_norm = 1e-4
    deform_grad = _get_deform_grad(displacement, dx_big)

    def body(state):
        rotation_inv_T = jnp.linalg.inv(state.rotation).transpose((0, 2, 1))
        rotation_new = 0.5 * (state.rotation + rotation_inv_T)
        norm = jnp.linalg.norm(state.rotation - rotation_new)
        iteration = state.iteration + 1
        return _GetRotationState(
            rotation=rotation_new, norm=norm, iteration=iteration, success=iteration < max_iter
        )

    state = _GetRotationState(rotation=deform_grad, norm=0, iteration=0, success=True)
    state = lax.while_loop(
        lambda state: (state.norm > max_norm) & (state.iteration < max_iter), body, state
    )
    final_rotation = jnp.linalg.inv(np.mean(state.rotation, axis=0))
    return final_rotation, state.success


class Dynamics(BodyPosition):
    def __init__(
        self,
        mesh_prop: MeshProperties,
        body_prop: StaticBodyProperties,
        schedule: Schedule,
        dynamics_config: DynamicsConfiguration,
        boundaries_description: Optional[BoundariesDescription] = None,
    ):
        super().__init__(
            mesh_prop=mesh_prop,
            schedule=schedule,
            boundaries_description=boundaries_description,
            create_in_subprocess=dynamics_config.create_in_subprocess,
        )
        self.body_prop = body_prop
        self.with_lhs = dynamics_config.with_lhs
        self.with_schur = dynamics_config.with_schur

        self.solver_cache = SolverMatrices()
        self.matrices = ConstMatrices()
        self.reinitialize_matrices()
        self.set_rotation()

    def set_displacement_old(self, displacement):
        super().set_displacement_old(displacement)
        self.set_rotation()

    def set_rotation(self):
        self.moved_base = self.get_rotation(self.displacement_old)

    def get_rotation(self, displacement):
        result = _get_rotation_jax(displacement, self.matrices.dx_big_jax)
        if not result[1]:
            raise Exception("Error calculating rotation")
        return complete_base(base_seed=np.array(result[0], dtype=np.float64))

    # def iterate_self(self, acceleration, temperature=None):
    #     super().iterate_self(acceleration, temperature)
    #     # self.reinitialize_matrices()  ###!!!

    def remesh(self, boundaries_description, create_in_subprocess):
        super().remesh(boundaries_description, create_in_subprocess)
        self.reinitialize_matrices()

    def reinitialize_matrices(self):
        # print("Initializing matrices...")
        def fun_dyn():
            return get_dynamics(
                elements=self.elements,
                nodes=self.moved_nodes,
                body_prop=self.body_prop,
                independent_indices=slice(self.nodes_count),  # self.independent_indices,
            )

        self.matrices = cmh.profile(fun_dyn, baypass=True)

        self.solver_cache.lhs_acceleration_jax = jxh.to_jax_sparse(
            self.matrices.acceleration_operator
        )

        if self.with_temperature:
            i = self.independent_indices

            self.solver_cache.lhs_temperature_sparse_jax = jxh.to_jax_sparse(
                (1 / self.time_step) * self.matrices.acceleration_operator[i, i]
                + self.matrices.thermal_conductivity[i, i]
            )

        if self.with_lhs:
            print("Creating LHS...")
            self.solver_cache.lhs_sparse = (
                self.matrices.acceleration_operator
                + (self.matrices.viscosity + self.matrices.elasticity * self.time_step)
                * self.time_step
            )

            self.solver_cache.lhs_sparse_jax = jxh.to_jax_sparse(self.solver_cache.lhs_sparse)
            # Calculating Jacobi preconditioner
            # TODO: Check SSOR / Incomplete Cholesky
            self.solver_cache.lhs_preconditioner_jax = jxh.to_jax_sparse(
                jxh.to_inverse_diagonal(self.solver_cache.lhs_sparse)
            )

        if self.with_schur:
            print("Creating Schur matrices...")
            # lhs_dense = self.solver_cache.lhs_sparse.todense()
            (
                self.solver_cache.contact_x_contact,
                self.solver_cache.free_x_contact,
                self.solver_cache.contact_x_free,
                self.solver_cache.free_x_free,
                self.solver_cache.lhs_boundary,
                self.solver_cache.free_x_free_inverted,
            ) = SchurComplement.calculate_schur_complement_matrices_jax(
                matrix=self.solver_cache.lhs_sparse,
                dimension=self.dimension,
                contact_indices=self.contact_indices,
                free_indices=self.free_indices,  ###
            )

            free_x_free_inverted = jax.scipy.linalg.inv(self.solver_cache.free_x_free.todense())
            self.solver_cache.lhs_boundary = (
                self.solver_cache.contact_x_contact.todense()
                - self.solver_cache.contact_x_free.todense()
                @ free_x_free_inverted
                @ self.solver_cache.free_x_contact.todense()
            )

            if self.with_temperature:
                (
                    self.solver_cache.temperature_boundary,
                    self.solver_cache.temperature_free_x_contact,
                    self.solver_cache.temperature_contact_x_free,
                    self.solver_cache.temperature_free_x_free,
                    self.solver_cache.temperature_free_x_free_inv,
                ) = SchurComplement.calculate_schur_complement_matrices_np(
                    matrix=self.solver_cache.lhs_temperature,
                    dimension=1,
                    contact_indices=self.contact_indices,
                    free_indices=self.free_indices,
                )

    @property
    def volume_at_nodes(self):
        return jxh.to_dense_np(self.matrices.volume_at_nodes)

    @property
    def with_temperature(self):
        return isinstance(self.body_prop, TemperatureBodyProperties)
