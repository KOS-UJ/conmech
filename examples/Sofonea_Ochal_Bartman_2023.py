# CONMECH @ Jagiellonian University in Kraków
#
# Copyright (C) 2023  Piotr Bartman <piotr.bartman@uj.edu.pl>
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

import numpy as np
from matplotlib import pyplot as plt

from conmech.helpers.config import Config
from conmech.mesh.boundaries_description import BoundariesDescription
from conmech.plotting.drawer import Drawer
from conmech.scenarios.problems import RelaxationQuasistaticProblem
from conmech.simulations.problem_solver import QuasistaticRelaxation

from examples.p_slope_contact_law import make_slope_contact_law
from examples.utils import elastic_relaxation_constitutive_law

eps = 1e-18


@dataclass
class QuasistaticSetup(RelaxationQuasistaticProblem):
    grid_height: ... = 1.0
    elements_number: ... = (2, 5)
    mu_coef: ... = 400
    la_coef: ... = 400
    time_step: ... = 0.1
    contact_law: ... = make_slope_contact_law(slope=5)

    relaxation: ... = np.array(
        [
            [[1200.0, 400.0, 400.0], [400.0, 400.0, 400.0], [400.0, 400.0, 400.0]],
            [[400.0, 400.0, 400.0], [400.0, 400.0, 1200.0], [400.0, 400.0, 400.0]],
        ]
    ) / 10

    @staticmethod
    def inner_forces(x, t=None):
        return np.array([0.0, 0.0])

    @staticmethod
    def outer_forces(x, t=None):
        return np.array([-0.2, -0.2])

    @staticmethod
    def friction_bound(u_nu):
        return 0

    boundaries: ... = BoundariesDescription(
        contact=lambda x: x[0] >= 4 and x[1] < eps, dirichlet=lambda x: x[0] <= 1 and x[1] < eps
    )


def main(show: bool = True, save: bool = False):
    ox = 2.5
    oy = 2.0
    eps = 0.001
    r_big = 2.5
    r_small = 1.5
    r = (r_big + r_small) / 2
    fv = 0.6
    left = 0
    right = 5

    def outer_forces(x, t):
        if x[1] <= oy:
            # if x[0] < left + eps:
            #     return np.array([0, -fv])
            # if x[0] > right - eps:
            #     return np.array([0, -fv])
            return np.array([0., 0])
        if (x[0] - ox) ** 2 + (x[1] - oy) ** 2 >= (r + eps) ** 2:
            return np.array([0, -fv])
        return np.array([0.0, 0.0])

    setup = QuasistaticSetup(mesh_type="tunnel")

    for outer_forces in (outer_forces,)[:]:
        setup.outer_forces = outer_forces

        runner = QuasistaticRelaxation(setup, solving_method="schur")

        states = runner.solve(
            n_steps=100,
            output_step=(0, 10, 25, 50, 100),
            verbose=False,
            initial_absement=setup.initial_absement,
            initial_displacement=setup.initial_displacement,
        )
        config = Config()
        absements = [state.absement for state in states]
        for i, state in enumerate(states):
            stress = elastic_relaxation_constitutive_law(
                state.displacement,
                absements[:i+1],
                setup,
                state.body.mesh.elements,
                state.body.mesh.initial_nodes,
            )
            c = np.linalg.norm(stress, axis=(1, 2))
            state.temperature = c  # stress[:, 0, 1]
            drawer = Drawer(state=state, config=config)
            drawer.node_size = 0
            drawer.original_mesh_color = None
            drawer.deformed_mesh_color = None
            drawer.cmap = plt.cm.rainbow
            drawer.x_min = 0
            drawer.x_max = 5
            drawer.y_min = 0
            drawer.y_max = 4.5
            if i == 0:
                drawer.outer_forces_scale = 1
            drawer.draw(show=True, title=f"time: {state.time:.2f}", temp_min=0, temp_max=30, save=False)


@dataclass
class OptimaControlProblem:
    desired_displacement: np.ndarray
    displacement_coef: float
    desired_inner_forces: np.ndarray
    inner_forces_coef: float
    desired_outer_forces: np.ndarray
    outer_forces_coef: float
    desired_foundation_rigidness: float
    foundation_rigidness_coef: float
    setup: QuasistaticSetup
    problem: QuasistaticRelaxation
    n_steps: int = 32

    def _cost(self, inner_forces, outer_forces, foundation_rigidness) -> float:
        self.setup.contact_law = make_slope_contact_law(slope=foundation_rigidness)
        self.problem.body._node_inner_forces = inner_forces
        self.problem.body._node_outer_forces = outer_forces
        self.problem.solving_method = "schur"
        states = self.problem.solve(
            n_steps=self.n_steps,
            output_step=range(self.n_steps),
            verbose=False,
            initial_absement=self.setup.initial_absement,
            initial_displacement=self.setup.initial_displacement,
        )
        displacement_norm = max([
            self.displacement_coef * np.linalg.norm(state.displacement - self.desired_displacement)
            for state in states
        ])

        cost = (displacement_norm
                + self.inner_forces_coef * np.linalg.norm(inner_forces - self.desired_inner_forces)
                + self.outer_forces_coef * np.linalg.norm(outer_forces - self.desired_outer_forces)
                + self.foundation_rigidness_coef * np.linalg.norm(
                    foundation_rigidness - self.desired_foundation_rigidness)
                )
        return cost

    def cost(self, arg: np.ndarray) -> float:
        start_1 = 2*len(self.desired_inner_forces)
        start_2 = start_1 + 2*len(self.desired_outer_forces)
        return self._cost(
            inner_forces=arg[:start_1].reshape(self.desired_inner_forces.shape),
            outer_forces=arg[start_1:start_2].reshape(self.desired_outer_forces.shape),
            foundation_rigidness=arg[-1]
        )


if __name__ == "__main__":
    main()
    # setup = QuasistaticSetup(mesh_type="tunnel")
    # problem = TimeDependentLongMemory(setup, solving_method="schur")
    # desired_displacement: np.ndarray = np.full_like(problem.body.mesh.initial_nodes, 0)
    # desired_inner_forces: np.ndarray = np.full_like(problem.body.inner_forces, 0)
    # desired_outer_forces: np.ndarray = np.full_like(problem.body.outer_forces, 0)
    # ocp = OptimaControlProblem(
    #     desired_displacement=desired_displacement,
    #     displacement_coef=1,
    #     desired_inner_forces=desired_inner_forces,
    #     inner_forces_coef=1,
    #     desired_outer_forces=desired_outer_forces,
    #     outer_forces_coef=1,
    #     desired_foundation_rigidness=0,
    #     foundation_rigidness_coef=1,
    #     setup=setup,
    #     problem=problem,
    # )
    # length = len(problem.body.inner_forces.ravel()) + len(problem.body.outer_forces.ravel()) + 1
    # initial_guess = np.zeros(length)
    # # for _ in range(100):
    # #     ocp.cost(initial_guess)
    # result = scipy.optimize.minimize(ocp.cost, initial_guess)
    # solution = result.x
