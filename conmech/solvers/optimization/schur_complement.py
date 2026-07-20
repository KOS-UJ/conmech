# CONMECH @ Jagiellonian University in Kraków
#
# Copyright (C) 2021-2026  Piotr Bartman-Szwarc <piotr.bartman@uj.edu.pl>
#
# This program is free software; you can redistribute it and/or
# modify it under the terms of the GNU General Public License
# as published by the Free Software Foundation; either version 3
# of the License, or (at your option) any later version.
#
# This program is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
# GNU General Public License for more details.
#
# You should have received a copy of the GNU General Public License
# along with this program; if not, write to the Free Software
# Foundation, Inc., 51 Franklin Street, Fifth Floor, Boston, MA  02110-1301,
# USA.

from typing import Tuple

import numpy as np
import scipy.sparse
import scipy.sparse.linalg

from conmech.dynamics.statement import Variables
from conmech.helpers import nph
from conmech.solvers._solvers import SolversRegistry
from conmech.solvers.optimization.optimization import Optimization


class FactorizedInverse:
    """Replacement for an explicit inverse of ``free_x_free``.

    Instead of ``inv(free_x_free)`` (dense) a sparse LU factorization is stored.
    The ``@`` operator mirrors ``inverse @ rhs`` so no other code has to know
    whether the inverse is dense sparse.
    """

    def __init__(self, matrix):
        # ``splu`` needs CSC
        self._solver = scipy.sparse.linalg.splu(scipy.sparse.csc_matrix(matrix))

    def __matmul__(self, other):
        dense = other.toarray() if scipy.sparse.issparse(other) else np.asarray(other)
        return self._solver.solve(dense)


class SchurComplementOptimization(Optimization):
    def __init__(
        self,
        statement,
        body,
        time_step,
        contact_law,
        driving_vector,
    ):
        super().__init__(
            statement,
            body,
            time_step,
            contact_law,
            driving_vector,
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
            dimension=self.statement.dimension_in,
            contact_indices=self.contact_ids,
            free_indices=self.free_ids,
        )

    def recalculate_forces(self):
        node_forces, forces_free = calculate_schur_complement_vector(
            vector=self.statement.right_hand_side,
            dimension=self.statement.dimension_in,
            contact_indices=self.contact_ids,
            free_indices=self.free_ids,
            contact_x_free=self.contact_x_free,
            free_x_free_inverted=self.free_x_free_inverted,
        )
        if self.statement.dimension_in == 2:
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
        **kwargs,
    ) -> np.ndarray:
        # initial_guess: [[xc, xf, xd] [yc, yf, yd]]
        truncated_ig = self.truncate_free_nodes(initial_guess)
        # truncated_ig: [[xc, yc]]
        if self.driving_vector:
            truncated_ig = truncated_ig[:, : truncated_ig.shape[1] // 2]
            # truncated_ig: [[xc]]
        solution_contact = super()._solve_impl(truncated_ig, **kwargs)
        # solution_contact: [xc, yc] / [xc]
        solution_free = self.complement_free_nodes(solution_contact)
        # solution_free: [[xf], [xd], [yf], [yd]] / [xf, xd]
        if self.driving_vector:
            length = len(solution_free)
            extender = np.zeros(length * 2).reshape(-1, 1)
            solution_free = solution_free.reshape(-1, 1)
            extender[:length, 0] = solution_free[:, 0]
            solution_free = extender

            length = len(solution_contact)
            extender = np.zeros(length * 2)
            extender[:length] = solution_contact[:]
            solution_contact = extender
            # solution_free: [[xf,], [xd,], [0,], [0,]]
            # solution_contact: [xc, 0]
        solution = self.merge(solution_contact, solution_free)
        # solution: [xc, xf, xd, yc, yf, yd]
        return solution

    def truncate_free_nodes(self, initial_guess: np.ndarray) -> np.ndarray:
        if self.statement.dimension_in == 2 or self.driving_vector:
            _result = initial_guess.reshape(2, -1)
            _result = _result[:, self.contact_ids]
            _result = _result.reshape(1, -1)
            result = _result
            return result
        return initial_guess[self.contact_ids]

    def complement_free_nodes(self, truncated_solution: np.ndarray) -> np.ndarray:
        if self.statement.dimension_in == 2:
            _result = truncated_solution.reshape(-1, 1)
        else:
            _result = truncated_solution

        _result = self.free_x_contact @ _result
        _result = self.forces_free - _result
        result = self.free_x_free_inverted @ _result
        return result

    def merge(self, solution_contact: np.ndarray, solution_free: np.ndarray) -> np.ndarray:
        if self.statement.dimension_in == 2 or self.driving_vector:
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


def _component_dofs(node_slice: slice, nodes_count: int, dimension: int) -> np.ndarray:
    nodes = np.arange(nodes_count)[node_slice]
    return np.concatenate([k * nodes_count + nodes for k in range(dimension)])


def calculate_schur_complement_matrices(
    matrix, dimension: int, contact_indices: slice, free_indices: slice
):
    nodes_count = matrix.shape[0] // dimension
    matrix = matrix.tocsr()

    free_dofs = _component_dofs(free_indices, nodes_count, dimension)
    contact_dofs = _component_dofs(contact_indices, nodes_count, dimension)

    def get_sliced(height_dofs, width_dofs):
        return matrix[height_dofs][:, width_dofs]

    free_x_free = get_sliced(free_dofs, free_dofs)
    free_x_contact = get_sliced(free_dofs, contact_dofs)
    contact_x_free = get_sliced(contact_dofs, free_dofs)
    contact_x_contact = get_sliced(contact_dofs, contact_dofs)

    free_x_free_inverted = FactorizedInverse(free_x_free)
    # The reduced (contact-sized) boundary matrix is dense on purpose.
    # It's small and goes into the optimization functional.
    matrix_boundary = np.asarray(
        contact_x_contact.toarray() - contact_x_free @ (free_x_free_inverted @ free_x_contact)
    )

    return matrix_boundary, free_x_contact, contact_x_free, free_x_free_inverted
