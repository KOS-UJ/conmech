"""
Created at 22.02.2021
"""

import math
from typing import Tuple

import numpy as np

from conmech.dynamics.statement import Variables
from conmech.helpers import nph
from conmech.solvers._solvers import SolversRegistry
from conmech.solvers.optimization.optimization import Optimization


class SchurComplementOptimization(Optimization):
    def __init__(
        self,
        statement,
        body,
        time_step,
        contact_law,
        friction_bound,
    ):
        super().__init__(
            statement,
            body,
            time_step,
            contact_law,
            friction_bound,
        )

        self.contact_ids = slice(0, body.mesh.contact_nodes_count)
        self.free_ids = slice(body.mesh.contact_nodes_count, body.mesh.nodes_count)

        (
            self._node_relations,
            self.free_x_contact,
            self.contact_x_free,
            self.free_x_free_inverted,
        ) = self.recalculate_displacement()

        self.node_forces_, self.forces_free = self.recalculate_forces()

    def recalculate_displacement(self):
        return calculate_schur_complement_matrices(
            matrix=self.statement.left_hand_side.data,
            dimension=self.statement.dimension,
            contact_indices=self.contact_ids,
            free_indices=self.free_ids,
        )

    def recalculate_forces(self):
        node_forces, forces_free = calculate_schur_complement_vector(
            vector=self.statement.right_hand_side,
            dimension=self.statement.dimension,
            contact_indices=self.contact_ids,
            free_indices=self.free_ids,
            contact_x_free=self.contact_x_free,
            free_x_free_inverted=self.free_x_free_inverted,
        )
        if self.statement.dimension == 2:
            return node_forces.T, forces_free
        return node_forces.reshape(-1), forces_free.reshape(-1)

    def __str__(self):
        return "schur"

    @property
    def lhs(self) -> np.ndarray:
        return self._node_relations

    @property
    def rhs(self) -> np.ndarray:
        return self.node_forces_

    def _solve_impl(
        self,
        initial_guess: np.ndarray,
        *,
        fixed_point_abs_tol: float = math.inf,
        **kwargs,
    ) -> np.ndarray:
        truncated_initial_guess = self.truncate_free_nodes(initial_guess)
        solution_contact = super()._solve_impl(
            truncated_initial_guess, fixed_point_abs_tol=fixed_point_abs_tol, **kwargs
        )
        solution_free = self.complement_free_nodes(solution_contact)
        solution = self.merge(solution_contact, solution_free)
        return solution

    def truncate_free_nodes(self, initial_guess: np.ndarray) -> np.ndarray:
        if self.statement.dimension == 2:
            _result = initial_guess.reshape(2, -1)
            _result = _result[:, self.contact_ids]
            _result = _result.reshape(1, -1)
            result = _result
            return result
        return initial_guess[self.contact_ids]

    def complement_free_nodes(self, truncated_solution: np.ndarray) -> np.ndarray:
        if self.statement.dimension == 2:
            _result = truncated_solution.reshape(-1, 1)
        else:
            _result = truncated_solution

        _result = self.free_x_contact @ _result
        _result = self.forces_free - _result
        result = self.free_x_free_inverted @ _result
        return result

    def merge(self, solution_contact: np.ndarray, solution_free: np.ndarray) -> np.ndarray:
        if self.statement.dimension == 2:
            u_contact = solution_contact.reshape(2, -1)
            u_free = solution_free.reshape(2, -1)
            _result = np.concatenate((u_contact, u_free), axis=1)
            _result = _result.reshape(1, -1)
            result = np.squeeze(np.asarray(_result))
            return result

        _result = np.concatenate((solution_contact, solution_free))
        result = np.squeeze(np.asarray(_result))
        return result


@SolversRegistry.register("static", "schur", "schur complement", "schur complement method")
class StaticSchurOptimization(SchurComplementOptimization):
    pass


@SolversRegistry.register("quasistatic", "schur", "schur complement", "schur complement method")
class QuasistaticSchurOptimization(SchurComplementOptimization):
    def iterate(self):
        self.statement.update(
            Variables(
                displacement=self.u_vector,
                time_step=self.time_step,
                time=self.current_time,
                electric_potential=self.p_vector,
            )
        )
        self.node_forces_, self.forces_free = self.recalculate_forces()


@SolversRegistry.register(
    "quasistatic relaxation", "schur", "schur complement", "schur complement method"
)
class QuasistaticRelaxed(SchurComplementOptimization):
    def iterate(self):
        self.statement.update(
            Variables(
                absement=self.b_vector,
                displacement=self.u_vector,
                time_step=self.time_step,
                time=self.current_time,
            )
        )
        self.node_forces_, self.forces_free = self.recalculate_forces()


@SolversRegistry.register("dynamic", "schur", "schur complement", "schur complement method")
class DynamicSchurOptimization(SchurComplementOptimization):
    def iterate(self):
        self.statement.update(
            Variables(
                displacement=self.u_vector,
                velocity=self.v_vector,
                temperature=self.t_vector,
                electric_potential=self.p_vector,
                time_step=self.time_step,
                time=self.current_time,
            )
        )
        self.node_forces_, self.forces_free = self.recalculate_forces()


def calculate_schur_complement_vector(
    vector: np.ndarray,
    dimension: int,
    contact_indices: slice,
    free_indices: slice,
    free_x_free_inverted: np.ndarray,
    contact_x_free: np.ndarray,
) -> Tuple[np.ndarray, np.ndarray]:
    vector_split = nph.unstack(vector, dimension)
    vector_contact = nph.stack_column(vector_split[contact_indices, :])
    vector_free = nph.stack_column(vector_split[free_indices, :])
    vector_boundary = vector_contact - (contact_x_free @ (free_x_free_inverted @ vector_free))
    return vector_boundary, vector_free


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
    matrix_boundary = contact_x_contact - contact_x_free @ (free_x_free_inverted @ free_x_contact)

    return matrix_boundary, free_x_contact, contact_x_free, free_x_free_inverted
