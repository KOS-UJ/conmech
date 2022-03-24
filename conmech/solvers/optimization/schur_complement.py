"""
Created at 22.02.2021
"""
import math
from typing import Tuple

import numpy as np
from conmech.forces import Forces
from conmech.helpers import nph
from conmech.solvers._solvers import Solvers
from conmech.solvers.optimization.optimization import Optimization


class SchurComplement(Optimization):
    def __init__(
        self,
        mesh,
        inner_forces,
        outer_forces,
        body_prop,
        time_step,
        contact_law,
        friction_bound,
    ):
        super().__init__(
            mesh,
            inner_forces,
            outer_forces,
            body_prop,
            time_step,
            contact_law,
            friction_bound,
        )

        self.contact_ids = slice(0, mesh.contact_count)
        self.free_ids = slice(mesh.contact_count, mesh.independent_nodes_count)
        n = self.mesh.independent_nodes_count

        # ADDED When working with velocity v, forces_contact depend on u

        C = self.get_C()

        (
            self._point_relations,
            self.free_x_contact,
            self.contact_x_free,
            self.free_x_free_inverted,
        ) = SchurComplement.calculate_schur_complement_matrices(
            matrix=C,
            dimension=self.mesh.dimension,
            contact_indices=self.contact_ids,
            free_indices=self.free_ids,
        )

        self.forces_free, self._point_forces = self.recalculate_forces()

    def recalculate_forces(self):
        E_split = self.get_E_split()
        # Et
        forces_contact = nph.stack_column(E_split[self.contact_ids, :])
        # Ei
        forces_free = nph.stack_column(E_split[self.free_ids, :])

        # Ebig = self.FVector - Bu + (1./self.tS) * ACCv / Ebig = self.FVector - X
        # Et = np.append(Ebig[self.i:self.n], Ebig[self.n + self.i:self.n + self.n])
        # self.Ei = np.append(Ebig[0:self.i], Ebig[self.n:self.n + self.i])

        # CiiINVEi = multiplyByDAT('E:\\SPARE\\cross ' + str(self.SizeH) + ' CiiINV.dat', self.Ei)
        _point_forces = self.free_x_free_inverted @ forces_free
        _point_forces = self.contact_x_free @ _point_forces
        # self.E = (Et - np.asarray(self.Cti.dot(CiiINVEi))).astype(np.single)
        _point_forces = forces_contact - _point_forces
        point_forces = np.asarray(_point_forces.reshape(1, -1))

        return forces_free, point_forces

    def get_C(self):
        raise NotImplementedError()

    def get_E_split(self):
        raise NotImplementedError()

    def __str__(self):
        return "schur"

    @staticmethod
    def calculate_schur_complement_matrices(
        matrix: np.ndarray, dimension: int, contact_indices: slice, free_indices: slice
    ):
        def get_sliced(matrix_split, indices_height, indices_width):
            matrix = np.moveaxis(matrix_split[..., indices_height, indices_width], 1, 2)
            dim, height, _, width = matrix.shape
            return matrix.reshape(dim * height, dim * width)

        matrix_split = np.array(
            np.split(np.array(np.split(matrix, dimension, axis=-1)), dimension, axis=1)
        )
        # Cii
        free_x_free = get_sliced(matrix_split, free_indices, free_indices)
        # Cit
        free_x_contact = get_sliced(matrix_split, free_indices, contact_indices)
        # Cti
        contact_x_free = get_sliced(matrix_split, contact_indices, free_indices)
        # Ctt
        contact_x_contact = get_sliced(matrix_split, contact_indices, contact_indices)

        # CiiINV
        free_x_free_inverted = np.linalg.inv(free_x_free)
        matrix_boundary = contact_x_contact - contact_x_free @ (
            free_x_free_inverted @ free_x_contact
        )

        return matrix_boundary, free_x_contact, contact_x_free, free_x_free_inverted


    @property
    def point_relations(self) -> np.ndarray:
        return self._point_relations

    @property
    def point_forces(self) -> np.ndarray:
        return self._point_forces

    def solve(
        self,
        initial_guess: np.ndarray,
        *,
        fixed_point_abs_tol: float = math.inf,
        **kwargs
    ) -> np.ndarray:
        truncated_initial_guess = self.truncate_free_points(initial_guess)
        solution_contact = super().solve(
            truncated_initial_guess, fixed_point_abs_tol=fixed_point_abs_tol, **kwargs
        )
        solution_free = self.complement_free_points(solution_contact)
        solution = self.merge(solution_contact, solution_free)
        return solution

    def solve_t(self, temperature, velocity) -> np.ndarray:
        truncated_initial_guess = self.truncate_free_points(velocity)
        truncated_temperature = temperature[self.contact_ids]
        solution_contact = super().solve_t(
            truncated_temperature, truncated_initial_guess[0]
        )  # reduce dim

        _solution_free = self.T_free_x_contact @ solution_contact
        _solution_free = self.Q_free - _solution_free
        solution_free = self.T_free_x_free_inverted @ _solution_free

        _result = np.concatenate((solution_contact, solution_free))
        solution = np.squeeze(np.asarray(_result))

        return solution

    def truncate_free_points(self, initial_guess: np.ndarray) -> np.ndarray:
        _result = initial_guess.reshape(2, -1)
        _result = _result[:, self.contact_ids]
        _result = _result.reshape(1, -1)
        result = _result
        return result

    def complement_free_points(self, truncated_solution: np.ndarray) -> np.ndarray:
        _result = truncated_solution.reshape(-1, 1)
        _result = self.free_x_contact @ _result
        _result = self.forces_free - _result
        result = self.free_x_free_inverted @ _result
        return result

    @staticmethod
    def merge(solution_contact: np.ndarray, solution_free: np.ndarray) -> np.ndarray:
        u_contact = solution_contact.reshape(2, -1)
        u_free = solution_free.reshape(2, -1)
        _result = np.concatenate((u_contact, u_free), axis=1)
        _result = _result.reshape(1, -1)
        result = np.squeeze(np.asarray(_result))
        return result


@Solvers.register("static", "schur", "schur complement", "schur complement method")
class Static(SchurComplement):
    def get_C(self):
        return self.B

    def get_E_split(self):
        return self.forces.F


@Solvers.register("quasistatic", "schur", "schur complement", "schur complement method")
class Quasistatic(SchurComplement):
    def __init__(
        self,
        mesh,
        inner_forces,
        outer_forces,
        body_prop,
        time_step,
        contact_law,
        friction_bound,
    ):
        self.A = mesh.A
        self.dim = mesh.dimension
        super().__init__(
            mesh,
            inner_forces,
            outer_forces,
            body_prop,
            time_step,
            contact_law,
            friction_bound,
        )

    def get_C(self):
        return self.A

    def get_E_split(self):
        return self.forces.F - nph.unstack(self.B @ self.u_vector.T, dim=self.dim)

    def iterate(self, velocity):
        super(SchurComplement, self).iterate(velocity)
        self.forces_free, self._point_forces = self.recalculate_forces()


@Solvers.register("dynamic", "schur", "schur complement", "schur complement method")
class Dynamic(Quasistatic):
    def __init__(
        self,
        mesh,
        inner_forces,
        outer_forces,
        body_prop,
        time_step,
        contact_law,
        friction_bound,
    ):
        self.dim = mesh.dimension
        self.ACC = mesh.ACC
        self.C2T = mesh.C2T
        self.K = mesh.K
        self.ind = mesh.independent_nodes_count
        self.t_vector = np.zeros(self.ind)
        super().__init__(
            mesh,
            inner_forces,
            outer_forces,
            body_prop,
            time_step,
            contact_law,
            friction_bound,
        )

        T = (1 / self.time_step) * self.ACC[: self.ind, : self.ind] + self.K[
            : self.ind, : self.ind
        ]

        (
            self._point_temperature,
            self.T_free_x_contact,
            self.T_contact_x_free,
            self.T_free_x_free_inverted,
        ) = SchurComplement.calculate_schur_complement_matrices(
            matrix=T,
            dimension=1,
            contact_indices=self.contact_ids,
            free_indices=self.free_ids,
        )

        # TODO #50
        # def inner_forces(x, y):
        #     return 0.1 * (1.25 - abs(x - 1.25) + 0.5 - abs(y - 0.5))
        #
        # def outer_forces(x, y):
        #     return 0
        #
        # self.inner_temperature = Forces(mesh, inner_forces, outer_forces)
        # self.inner_temperature.setF()

        self.Q_free, self.Q = self.recalculate_temperature()

    @property
    def T(self):
        return self._point_temperature

    def get_C(self):
        return self.A + (1 / self.time_step) * self.ACC

    def get_E_split(self):
        X = -1 * self.B @ self.u_vector

        X += (1 / self.time_step) * self.ACC @ self.v_vector

        X += np.tile(self.t_vector, self.dim) @ self.C2T  # TODO: Check if not -1 *

        return self.forces.F + nph.unstack(X, dim=self.dim)

    def iterate(self, velocity):
        super(SchurComplement, self).iterate(velocity)
        self.forces_free, self._point_forces = self.recalculate_forces()
        self.Q_free, self.Q = self.recalculate_temperature()

    def recalculate_temperature(self):
        QBig = (-1) * nph.unstack_and_sum_columns(
            self.C2T @ self.v_vector, dim=self.dim
        )

        QBig += (1 / self.time_step) * self.ACC[: self.ind, : self.ind] @ self.t_vector
        # QBig = self.inner_temperature.F[:, 0] + Q1 - C2Xv - C2Yv  # TODO #50

        Q_free = QBig[self.free_ids]
        Q_contact = QBig[self.contact_ids]
        # TiiINVQi = multiplyByDAT(prefix + ' TiiINV.dat', self.Qi)
        _point_temperature = self.T_free_x_free_inverted @ Q_free
        Q = Q_contact - np.asarray(self.T_contact_x_free.dot(_point_temperature))

        return Q_free, Q
