# CONMECH @ Jagiellonian University in Kraków
#
# Copyright (C) 2023  Piotr Bartman-Szwarc <piotr.bartman@uj.edu.pl>
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
from dataclasses import dataclass
from typing import Optional

import numpy as np
from conmech.helpers.config import Config
from conmech.mesh.boundaries_description import BoundariesDescription
from conmech.plotting.drawer import Drawer
from conmech.scenarios.problems import WaveProblem
from conmech.simulations.problem_solver import WaveSolver
from conmech.properties.mesh_description import CrossMeshDescription


@dataclass()
class MembraneSetup(WaveProblem):
    time_step: ... = 0.1
    propagation: ... = 1.0

    @staticmethod
    def inner_forces(x: np.ndarray, t: Optional[float] = None) -> np.ndarray:
        return np.array([0.2])

    @staticmethod
    def outer_forces(
        x: np.ndarray, v: Optional[np.ndarray] = None, t: Optional[float] = None
    ) -> np.ndarray:
        return np.array([0.0])

    boundaries: ... = BoundariesDescription(dirichlet=lambda x: x[0] in (0, 1) or x[1] in (0, 1))


def main(config: Config):
    """
    Entrypoint to example.

    To see result of simulation you need to call from python `main(Config().init())`.
    """
    max_element_perimeter = 1 / 8 if not config.test else 1 / 3
    mesh_descr = CrossMeshDescription(
        initial_position=None,
        max_element_perimeter=max_element_perimeter,
        scale=[1, 1],
    )
    setup = MembraneSetup(mesh_descr)
    runner = WaveSolver(setup, "direct")
    n_steps = 32 if not config.test else 3

    states = runner.solve(
        n_steps=n_steps,
        output_step=(0, n_steps),
        initial_displacement=setup.initial_displacement,
        initial_velocity=setup.initial_velocity,
        verbose=True,
    )
    drawer = Drawer(state=states[-1], config=config)
    drawer.draw(
        show=config.show,
        save=config.save,
        foundation=False,
    )


if __name__ == "__main__":
    main(Config().init())
