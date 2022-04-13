"""
Created at 22.02.2021
"""
import math
import numpy as np

from conmech.dynamics.statement import (
    StaticStatement,
    QuasistaticStatement,
    DynamicStatement,
    TemperatureStatement,
    Variables,
)
from conmech.helpers import nph
from conmech.solvers._solvers import Solvers
from conmech.solvers.optimization.optimization import Optimization


class SchurComplement(Optimization):
    def __init__(
        self,
        mesh,
        statement,
        body_prop,
        time_step,
        contact_law,
        friction_bound,
    ):
        super().__init__(
            mesh,
            statement,
            body_prop,
            time_step,
            contact_law,
            friction_bound,
        )

        self.contact_ids = slice(0, mesh.contact_nodes_count)
        self.free_ids = slice(mesh.contact_nodes_count, mesh.independent_nodes_count)

        (
            self._point_relations,
            self.free_x_contact,
            self.contact_x_free,
            self.free_x_free_inverted,
        ) = self.recalculate_displacement()

        self._point_forces, self.forces_free = self.recalculate_forces()

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
        free_x_free = get_sliced(matrix_split, free_indices, free_indices)
        free_x_contact = get_sliced(matrix_split, free_indices, contact_indices)
        contact_x_free = get_sliced(matrix_split, contact_indices, free_indices)
        contact_x_contact = get_sliced(matrix_split, contact_indices, contact_indices)

        free_x_free_inverted = np.linalg.inv(free_x_free)
        matrix_boundary = contact_x_contact - contact_x_free @ (
            free_x_free_inverted @ free_x_contact
        )

        return matrix_boundary, free_x_contact, contact_x_free, free_x_free_inverted

    @staticmethod
    def calculate_schur_complement_vector(
        vector: np.ndarray,
        dimension: int,
        contact_indices: slice,
        free_indices: slice,
        free_x_free_inverted: np.ndarray,
        contact_x_free: np.ndarray,
    ):
        vector_split = nph.unstack(vector, dimension)
        vector_contact = nph.stack_column(vector_split[contact_indices, :])
        vector_free = nph.stack_column(vector_split[free_indices, :])
        vector_boundary = vector_contact - (contact_x_free @ (free_x_free_inverted @ vector_free))
        return vector_boundary, vector_free

    def recalculate_displacement(self):
        return SchurComplement.calculate_schur_complement_matrices(
            matrix=self.statement.left_hand_side,
            dimension=self.mesh.dimension,
            contact_indices=self.contact_ids,
            free_indices=self.free_ids,
        )

    def recalculate_forces(self):
        point_forces, forces_free = SchurComplement.calculate_schur_complement_vector(
            vector=self.statement.right_hand_side,
            dimension=self.mesh.dimension,
            contact_indices=self.contact_ids,
            free_indices=self.free_ids,
            contact_x_free=self.contact_x_free,
            free_x_free_inverted=self.free_x_free_inverted,
        )
        return point_forces.T, forces_free

    def __str__(self):
        return "schur"

    @property
    def point_relations(self) -> np.ndarray:
        return self._point_relations

    @property
    def point_forces(self) -> np.ndarray:
        return self._point_forces

    def solve(
        self, initial_guess: np.ndarray, *, fixed_point_abs_tol: float = math.inf, **kwargs
    ) -> np.ndarray:
        truncated_initial_guess = self.truncate_free_points(initial_guess)
        solution_contact = super().solve(
            truncated_initial_guess, fixed_point_abs_tol=fixed_point_abs_tol, **kwargs
        )
        solution_free = self.complement_free_points(solution_contact)
        solution = self.merge(solution_contact, solution_free)
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
    def __init__(self, mesh, body_prop, time_step, contact_law, friction_bound):
        self.statement = StaticStatement(mesh)
        super().__init__(
            mesh,
            self.statement,
            body_prop,
            time_step,
            contact_law,
            friction_bound,
        )


@Solvers.register("quasistatic", "schur", "schur complement", "schur complement method")
class Quasistatic(SchurComplement):
    def __init__(
        self,
        mesh,
        body_prop,
        time_step,
        contact_law,
        friction_bound,
    ):
        self.statement = QuasistaticStatement(mesh)
        super().__init__(
            mesh,
            self.statement,
            body_prop,
            time_step,
            contact_law,
            friction_bound,
        )

    def iterate(self, velocity):
        super().iterate(velocity)
        self.statement.update(Variables(displacement=self.u_vector))
        self._point_forces, self.forces_free = self.recalculate_forces()


@Solvers.register("dynamic", "schur", "schur complement", "schur complement method")
class Dynamic(SchurComplement):
    def __init__(
        self,
        mesh,
        body_prop,
        time_step,
        contact_law,
        friction_bound,
    ):
        self.statement = DynamicStatement(mesh)
        self.temperature_statement = TemperatureStatement(mesh)
        super().__init__(
            mesh,
            self.statement,
            body_prop,
            time_step,
            contact_law,
            friction_bound,
        )
        self.temperature_statement.update(
            Variables(
                velocity=self.v_vector,
                temperature=self.t_vector,
                time_step=self.time_step,
            )
        )

        (
            self._point_temperature,
            self.temper_free_x_contact,
            self.temper_contact_x_free,
            self.temper_free_x_free_inverted,
        ) = SchurComplement.calculate_schur_complement_matrices(
            matrix=self.temperature_statement.left_hand_side,
            dimension=1,
            contact_indices=self.contact_ids,
            free_indices=self.free_ids,
        )

        # TODO #50
        # def inner_forces(x):
        #     return 0.1 * (1.25 - abs(x - 1.25) + 0.5 - abs(y - 0.5))
        #
        # def outer_forces(x):
        #     return 0
        #
        # self.inner_temperature = Forces(mesh, inner_forces, outer_forces)
        # self.inner_temperature.setF()

        self.temper_rhs, self.temper_rhs_free = self.recalculate_temperature()

    # def solve(
    #     self,
    #     state,
    #     *,
    #     fixed_point_abs_tol: float = math.inf,
    #     **kwargs
    # ):
    #     velocity = super(Dynamic, self).solve(state["velocity"],
    #                                           fixed_point_abs_tol=fixed_point_abs_tol,
    #                                           **kwargs)
    #     state.set_velocity(velocity_vector=velocity)

    def solve_t(self, initial_guess, velocity) -> np.ndarray:
        truncated_initial_guess = self.truncate_free_points(velocity)
        truncated_temperature = initial_guess[self.contact_ids]
        solution_contact = super().solve_t(truncated_temperature, truncated_initial_guess[0])

        _solution_free = self.temper_free_x_contact @ solution_contact
        _solution_free = self.temper_rhs_free - _solution_free
        solution_free = self.temper_free_x_free_inverted @ _solution_free

        _result = np.concatenate((solution_contact, solution_free))
        solution = np.squeeze(np.asarray(_result))

        return solution

    @property
    def node_temperature(self):
        return self._point_temperature

    def iterate(self, velocity):
        super().iterate(velocity)
        self.statement.update(
            Variables(
                displacement=self.u_vector,
                velocity=self.v_vector,
                temperature=self.t_vector,
                time_step=self.time_step,
            )
        )
        self.temperature_statement.update(
            Variables(
                velocity=self.v_vector,
                temperature=self.t_vector,
                time_step=self.time_step,
            )
        )
        self._point_forces, self.forces_free = self.recalculate_forces()
        self.temper_rhs, self.temper_rhs_free = self.recalculate_temperature()

    def recalculate_temperature(self):
        A_contact, A_free = SchurComplement.calculate_schur_complement_vector(
            vector=self.temperature_statement.right_hand_side,
            dimension=1,
            contact_indices=self.contact_ids,
            free_indices=self.free_ids,
            contact_x_free=self.temper_contact_x_free,
            free_x_free_inverted=self.temper_free_x_free_inverted,
        )
        return A_contact.reshape(-1), A_free.reshape(-1)
