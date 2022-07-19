from dataclasses import dataclass
from typing import Callable, Optional

import jax.experimental.sparse
import jax.interpreters.xla
import jax.numpy as jnp
import jax.scipy
import numba
import numpy as np
import scipy.sparse

from conmech.dynamics.factory.dynamics_factory_method import ConstMatrices, get_dynamics
from conmech.helpers import cmh, jxh
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

        self.lhs_inv: jax.interpreters.xla.DeviceArray
        self.lhs_boundary: jax.interpreters.xla.DeviceArray

        self.lhs_temperature_sparse: jax.experimental.sparse.BCOO  # sparse.base.spmatrix
        self.temperature_boundary: np.ndarray
        self.temperature_free_x_contact: np.ndarray
        self.temperature_contact_x_free: np.ndarray
        self.temperature_free_x_free_inv: np.ndarray

    @property
    def lhs(self):
        return jxh.to_dense_np(self.lhs_sparse)

    @property
    def lhs_temperature(self):
        return jxh.to_dense_np(self.lhs_temperature_sparse)


@dataclass
class DynamicsConfiguration:
    normalize_by_rotation: bool = True
    create_in_subprocess: bool = False
    with_lhs: bool = True
    with_schur: bool = True


class Dynamics(BodyPosition):
    def __init__(
        self,
        mesh_prop: MeshProperties,
        body_prop: StaticBodyProperties,
        schedule: Schedule,
        dynamics_config: DynamicsConfiguration,
        is_dirichlet: Optional[Callable],
        is_contact: Optional[Callable],
    ):
        super().__init__(
            mesh_prop=mesh_prop,
            schedule=schedule,
            normalize_by_rotation=dynamics_config.normalize_by_rotation,
            is_dirichlet=is_dirichlet,
            is_contact=is_contact,
            create_in_subprocess=dynamics_config.create_in_subprocess,
        )
        self.body_prop = body_prop
        self.with_lhs = dynamics_config.with_lhs
        self.with_schur = dynamics_config.with_schur

        self.solver_cache = SolverMatrices()
        self.matrices = ConstMatrices()
        self.reinitialize_matrices()

    def iterate_self(self, acceleration, temperature=None):
        super().iterate_self(acceleration, temperature)
        # self.reinitialize_matrices()  ###!!!

    def remesh(self, is_dirichlet, is_contact, create_in_subprocess):
        super().remesh(is_dirichlet, is_contact, create_in_subprocess)
        self.reinitialize_matrices()

    def reinitialize_matrices(self):
        # print("Initializing matrices...")
        fun_dyn = lambda: get_dynamics(
            elements=self.elements,
            nodes=self.moved_nodes,
            body_prop=self.body_prop,
            independent_indices=self.independent_indices,
        )
        self.matrices = cmh.profile(fun_dyn, baypass=True)

        self.solver_cache.lhs_acceleration_jax = jxh.to_jax_sparse(self.matrices.acceleration_operator)

        if not self.with_lhs:
            return

        self.solver_cache.lhs_sparse = (
            self.matrices.acceleration_operator
            + (self.matrices.viscosity + self.matrices.elasticity * self.time_step) * self.time_step
        )

        lhs_sparse_cp = jxh.to_cupy_csr(self.solver_cache.lhs_sparse)
        self.solver_cache.lhs_sparse_cp = lhs_sparse_cp
        # Calculating Jacobi preconditioner
        # TODO: Check SSOR / Incomplete Cholesky
        self.solver_cache.lhs_preconditioner_cp = jxh.to_inverse_diagonal(lhs_sparse_cp)
        # ilu = cupyx.scipy.sparse.linalg.spilu(A=A, fill_factor=1)
        # M = cupyx.scipy.sparse.linalg.LinearOperator(A.shape, ilu.solve)
        # M = cupy.linalg.inv(A.todense())

        # print("Inverting lhs...")
        # lhs_dense = self.solver_cache.lhs_sparse.todense()
        # self.solver_cache.lhs_inv = jax.scipy.linalg.inv(lhs_dense)
        return
        if not self.with_schur:
            return

        print("Creating Schur matrices...")
        # lhs_dense = self.solver_cache.lhs_sparse.todense()
        (
            self.solver_cache.contact_x_contact,
            self.solver_cache.free_x_contact,
            self.solver_cache.contact_x_free,
            self.solver_cache.free_x_free,
            self.solver_cache.lhs_boundary,
            self.solver_cache.free_x_free_inverted,
        ) = SchurComplement.calculate_schur_complement_matrices(
            matrix=self.solver_cache.lhs_sparse_cp,
            dimension=self.dimension,
            contact_indices=self.contact_indices,
            free_indices=self.free_indices,
        )

        free_x_free_inverted = jax.scipy.linalg.inv(self.solver_cache.free_x_free.todense())
        self.solver_cache.lhs_boundary = (
            self.solver_cache.contact_x_contact.todense()
            - self.solver_cache.contact_x_free.todense()
            @ free_x_free_inverted
            @ self.solver_cache.free_x_contact.todense()
        )

        return
        if not self.with_temperature:
            return

        i = self.independent_indices
        # TODO: #65 Make faster
        self.solver_cache.lhs_temperature_sparse = jax.experimental.sparse.BCOO.fromdense(
            (1 / self.time_step) * self.acceleration_operator.todense()[i, i]
            + self.thermal_conductivity.todense()[i, i]
        )
        (
            self.solver_cache.temperature_boundary,
            self.solver_cache.temperature_free_x_contact,
            self.solver_cache.temperature_contact_x_free,
            self.solver_cache.temperature_free_x_free_inv,
        ) = SchurComplement.calculate_schur_complement_matrices(
            matrix=self.solver_cache.lhs_temperature,
            dimension=1,
            contact_indices=self.contact_indices,
            free_indices=self.free_indices,
        )

    @property
    def volume_at_nodes(self):
        return jxh.to_dense_np(self.matrices.volume_at_nodes)

    @property
    def acceleration_operator(self):
        return jxh.to_dense_np(self.acceleration_operator)

    @property
    def elasticity(self):
        return jxh.to_dense_np(self.elasticity)

    @property
    def viscosity(self):
        return jxh.to_dense_np(self.viscosity)

    @property
    def thermal_expansion(self):
        return jxh.to_dense_np(self.thermal_expansion)

    @property
    def thermal_conductivity(self):
        return jxh.to_dense_np(self.thermal_conductivity)

    @property
    def with_temperature(self):
        return isinstance(self.body_prop, TemperatureBodyProperties)
