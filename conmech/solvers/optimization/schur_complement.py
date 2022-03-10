"""
Created at 22.02.2021
"""
import math
from typing import Tuple

import numpy as np
from conmech.solvers._solvers import Solvers
from conmech.solvers.optimization.optimization import Optimization
from conmech.helpers import nph


class SchurComplement(Optimization):
    def __init__(
        self,
        mesh,
        inner_forces,
        outer_forces,
        coefficients,
        time_step,
        contact_law,
        friction_bound,
    ):
        super().__init__(
            mesh,
            inner_forces,
            outer_forces,
            coefficients,
            time_step,
            contact_law,
            friction_bound,
        )

        self.contact_ids = slice(0, mesh.contact_count)
        self.free_ids = slice(mesh.contact_count, mesh.independent_nodes_count)
        n = self.mesh.independent_nodes_count

        # ADDED When working with velocity v, forces_contact depend on u

        C = self.get_C()

        # Cii
        free_x_free = self.get_submatrix(
            C, indices=(self.free_ids, self.free_ids), ind_num=n
        )
        # Cit
        free_x_contact = self.get_submatrix(
            C, indices=(self.free_ids, self.contact_ids), ind_num=n
        )
        # Cti
        self.contact_x_free = self.get_submatrix(
            C, indices=(self.contact_ids, self.free_ids), ind_num=n
        )
        # Ctt
        contact_x_contact = self.get_submatrix(
            C, indices=(self.contact_ids, self.contact_ids), ind_num=n
        )

        self.free_x_contact = free_x_contact
        # CiiINV:
        self.free_x_free_inverted = np.linalg.inv(free_x_free)
        # CiiINVCit:
        _point_relations = self.free_x_free_inverted @ self.free_x_contact
        # CtiCiiINVCit:
        _point_relations = self.contact_x_free @ _point_relations
        # Ctt - CtiCiiINVCit:
        _point_relations = contact_x_contact - _point_relations
        self._point_relations = np.asarray(_point_relations)
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
    def get_submatrix(
        arrays: iter, indices: Tuple[slice, slice], ind_num: int
    ) -> np.matrix:
        ind00 = (slice(0, ind_num), slice(0, ind_num))
        ind01 = (slice(0, ind_num), slice(ind_num, 2 * ind_num))
        ind10 = (slice(ind_num, 2 * ind_num), slice(0, ind_num))
        ind11 = (slice(ind_num, 2 * ind_num), slice(ind_num, 2 * ind_num))
        result = np.bmat(
            [
                [arrays[ind00][indices], arrays[ind01][indices]],
                [arrays[ind10][indices], arrays[ind11][indices]],
            ]
        )
        return result

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
        temperature=None,
        fixed_point_abs_tol: float = math.inf,
        **kwargs
    ) -> np.ndarray:
        truncated_initial_guess = self.truncate_free_points(initial_guess)
        truncated_temperature = None
        if temperature is not None:
            truncated_temperature = temperature[self.contact_ids]
        solution_contact = super().solve(
            truncated_initial_guess,
            temperature=truncated_temperature,
            fixed_point_abs_tol=fixed_point_abs_tol,
            **kwargs
        )
        solution_free = self.complement_free_points(solution_contact)
        solution = self.merge(solution_contact, solution_free)
        self.iterate(solution)
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

        self.t_vector = solution
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
        coefficients,
        time_step,
        contact_law,
        friction_bound,
    ):
        self.A = mesh.A
        super().__init__(
            mesh,
            inner_forces,
            outer_forces,
            coefficients,
            time_step,
            contact_law,
            friction_bound,
        )

    def get_C(self):
        return self.A

    def get_E_split(self):
        return self.forces.F - nph.unstack(self.B @ self.u_vector.T, dim=2)

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
        coefficients,
        time_step,
        contact_law,
        friction_bound,
    ):
        self.ACC = mesh.ACC
        self.K = mesh.K
        self.t_vector = np.zeros(mesh.independent_nodes_count)
        super().__init__(
            mesh,
            inner_forces,
            outer_forces,
            coefficients,
            time_step,
            contact_law,
            friction_bound,
        )

        T = (1 / self.time_step) \
            * self.ACC[:self.mesh.independent_nodes_count, :self.mesh.independent_nodes_count] \
            + self.K[:self.mesh.independent_nodes_count, :self.mesh.independent_nodes_count]

        # Tii
        T_free_x_free = T[self.free_ids, self.free_ids]
        # Tit
        self.T_free_x_contact = T[self.free_ids, self.contact_ids]
        # Tti
        self.T_contact_x_free = T[self.contact_ids, self.free_ids]
        # Ttt
        T_contact_x_contact = T[self.contact_ids, self.contact_ids]

        # TiiINV:
        self.T_free_x_free_inverted = np.linalg.inv(T_free_x_free)
        # TiiINVCit:
        _point_temperature = np.dot(self.T_free_x_free_inverted, self.T_free_x_contact)
        # TtiTiiINVTit:
        _point_temperature = np.dot(self.T_contact_x_free, _point_temperature)
        # Ttt - TtiTiiINVTit:
        _point_temperature = T_contact_x_contact - _point_temperature
        self._point_temperature = np.asarray(_point_temperature)

        self.Q_free, self.Q = self.recalculate_temperature()

    @property
    def T(self):
        return self._point_temperature

    def get_C(self):
        return self.A + (1 / self.time_step) * self.ACC

    def get_E_split(self):
        X = -1 * nph.unstack(self.B @ self.u_vector, dim=2)

        X += (1 / self.time_step) * nph.unstack(self.ACC @ self.v_vector, dim=2)

        C2X, C2Y = self.mesh.C2X, self.mesh.C2Y
        C2XTemp = np.squeeze(np.dot(np.transpose(C2X), self.t_vector[0:self.mesh.independent_nodes_count].transpose()))
        C2YTemp = np.squeeze(np.dot(np.transpose(C2Y), self.t_vector[0:self.mesh.independent_nodes_count].transpose()))

        X += -1 * np.stack((C2XTemp, C2YTemp), axis=-1)

        return self.forces.F + X

    def iterate(self, velocity):
        super(SchurComplement, self).iterate(velocity)
        self.forces_free, self._point_forces = self.recalculate_forces()
        self.Q_free, self.Q = self.recalculate_temperature()

    def recalculate_temperature(self):
        C2X, C2Y = self.mesh.C2X, self.mesh.C2Y

        C2Xv = np.squeeze(
            np.asarray(
                C2X @ self.v_vector[0: self.mesh.independent_nodes_count].transpose(),
            )
        )
        C2Yv = np.squeeze(
            np.asarray(
                C2Y @ self.v_vector[
                    self.mesh.independent_nodes_count: 2 * self.mesh.independent_nodes_count
                ].transpose()
            )
        )

        Q1 = (1 / self.time_step) * np.squeeze(
            np.asarray(
                self.ACC[:self.mesh.independent_nodes_count, :self.mesh.independent_nodes_count]
                @ self.t_vector[:self.mesh.independent_nodes_count].transpose(),
            )
        )

        QBig = Q1 + C2Xv + C2Yv ###############################

        Q_free = QBig[self.free_ids]
        Q_contact = QBig[self.contact_ids]
        # TiiINVQi = multiplyByDAT(prefix + ' TiiINV.dat', self.Qi)
        _point_temperature = self.T_free_x_free_inverted @ Q_free
        Q = Q_contact - np.asarray(self.T_contact_x_free.dot(_point_temperature))

        return Q_free, Q