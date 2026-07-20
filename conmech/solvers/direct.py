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

from typing import Optional, Callable

import numpy as np
import scipy.optimize
import scipy.sparse.linalg

from conmech.dynamics.statement import Statement
from conmech.dynamics.contact.contact_law import DirectContactLaw
from conmech.scene.body_forces import BodyForces
from conmech.solvers._solvers import SolversRegistry
from conmech.solvers.solver import Solver
from conmech.solvers.solver_methods import make_equation


@SolversRegistry.register("dynamic", "direct")
@SolversRegistry.register("static", "direct")
class Direct(Solver):
    def __init__(
        self,
        statement: Statement,
        body: BodyForces,
        time_step: float,
        contact_law: Optional[DirectContactLaw] = None,
        driving_vector: bool = False,
    ):
        super().__init__(
            statement,
            body,
            time_step,
            contact_law,
            driving_vector,
        )
        self.equation: Optional[Callable] = None

        if contact_law is not None:
            self.equation = make_equation(
                jn=contact_law.subderivative_normal_direction,
                contact=(
                    contact_law.general_contact_condition
                    if hasattr(contact_law, "general_contact_condition")
                    else None
                ),
            )

    def __str__(self) -> str:
        return "direct"

    @property
    def node_relations(self) -> np.ndarray:
        return self.statement.left_hand_side.data

    @property
    def node_forces(self) -> np.ndarray:
        return self.statement.right_hand_side

    def _solve_impl(
        self,
        initial_guess: np.ndarray,
        *,
        variable_old: np.ndarray,
        displacement: np.ndarray,
        **kwargs,
    ) -> np.ndarray:
        displacement = np.squeeze(displacement.copy().reshape(1, -1))
        if self.equation is not None:
            result = scipy.optimize.fsolve(
                self.equation,
                initial_guess,
                args=(
                    variable_old,
                    self.body.mesh.nodes,
                    self.body.mesh.contact_boundary,
                    self.body.mesh.boundaries.contact_normals,
                    np.ascontiguousarray(self.node_relations.todense()),
                    self.node_forces,
                    displacement,
                    self.body.dynamics.acceleration_operator.SM1.bare,
                    self.time_step,
                ),
            )
        else:
            result = scipy.sparse.linalg.spsolve(self.node_relations, self.node_forces)
            result_len = len(result)
            var_len = len(initial_guess.ravel())
            if result_len < var_len:
                result_ = np.zeros(var_len)
                result_[:result_len] = result[:]
                result = result_
        return np.asarray(result)
