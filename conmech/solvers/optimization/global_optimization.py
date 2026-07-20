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

import numpy as np

from conmech.dynamics.statement import Variables
from conmech.solvers._solvers import SolversRegistry
from conmech.solvers.optimization.optimization import Optimization


class GlobalOptimization(Optimization):
    def __str__(self):
        return "global optimization"

    @property
    def lhs(self) -> np.ndarray:
        return self.statement.left_hand_side.bare

    @property
    def rhs(self) -> np.ndarray:
        return self.statement.right_hand_side


@SolversRegistry.register("static", "global", "global optimization")
class StaticGlobalOptimization(GlobalOptimization):
    pass


@SolversRegistry.register("quasistatic", "global", "global optimization")
class QuasistaticGlobalOptimization(GlobalOptimization):
    def iterate(self):
        self.statement.update(
            Variables(
                displacement=self.u_vector,
                electric_potential=self.p_vector,
                time_step=self.time_step,
                time=self.current_time,
            )
        )


@SolversRegistry.register("quasistatic relaxation", "global", "global optimization")
class QuasistaticRelaxedGlobalOptimization(GlobalOptimization):
    def iterate(self):
        self.statement.update(
            Variables(
                absement=self.b_vector,
                displacement=self.u_vector,
                time_step=self.time_step,
            )
        )


@SolversRegistry.register("dynamic", "global", "global optimization")
class DynamicGlobalOptimization(GlobalOptimization):
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
