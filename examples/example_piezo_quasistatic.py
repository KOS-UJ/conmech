"""
Created at 21.08.2019
"""
from dataclasses import dataclass

import numpy as np

from conmech.helpers.config import Config
from conmech.mesh.boundaries_description import BoundariesDescription
from conmech.simulations.problem_solver import PiezoelectricTimeDependentSolver
from conmech.scenarios.problems import PiezoelectricQuasistaticProblem
from conmech.plotting.drawer import Drawer

from examples.p_slope_contact_law import make_slope_contact_law


class PPSlopeContactLaw(make_slope_contact_law(slope=1e1)):
    @staticmethod
    def h_nu(uN, t):
        return 0

    @staticmethod
    def h_tau(uN, t):
        return 0

    @staticmethod
    def h_temp(u_tau):  # potential  # TODO # 48
        return 0


@dataclass()
class PQuasistaticSetup(PiezoelectricQuasistaticProblem):
    grid_height: ... = 1.0
    elements_number: ... = (3, 4)
    mu_coef: ... = 4
    la_coef: ... = 4
    th_coef: ... = 4
    ze_coef: ... = 4
    time_step: ... = 0.001
    contact_law: ... = PPSlopeContactLaw
    piezoelectricity: ... = np.array([[0.5, 0.0, 0.0], [0.0, 0.5, 0.0], [0.0, 0.0, 0.5]])
    permittivity: ... = np.array([[0.1, 0.0, 0.0], [0.0, 0.1, 0.0], [0.0, 0.0, 0.1]])

    @staticmethod
    def initial_electric_potential(x: np.ndarray) -> np.ndarray:
        return np.asarray([0.0])

    @staticmethod
    def inner_forces(x):
        return np.array([0.0, 0.0])

    @staticmethod
    def outer_forces(x):
        return np.array([0, 0])

    @staticmethod
    def friction_bound(u_nu):
        return 0

    boundaries: ... = BoundariesDescription(
        contact=lambda x: 0.0 <= x[0] <= 1.0 and 0.0 <= x[1] <= 1.0,
        dirichlet=lambda x: 1.0 <= x[0] <= 1.5 and 1.0 <= x[1] <= 1.5,
        dirichlet_electric=lambda x: (
            x[0] == 1.0 and 1.0 <= x[1] <= 4.0 or 1.5 <= x[0] <= 3.0 and x[1] == 1.0
        ),
    )


def main(show: bool):
    setup = PQuasistaticSetup(mesh_type="Barboteu2008")
    runner = PiezoelectricTimeDependentSolver(setup, solving_method="schur")

    states = runner.solve(
        n_steps=32,
        output_step=range(0, 32, 4),
        verbose=True,
        initial_displacement=setup.initial_displacement,
        initial_velocity=setup.initial_velocity,
        initial_electric_potential=setup.initial_electric_potential,
    )
    e_max = -np.inf
    e_min = np.inf
    for state in states:
        e_max = max(e_max, np.max(state.electric_potential))
        e_min = min(e_min, np.min(state.electric_potential))
    config = Config()
    for state in states:
        Drawer(state=state, config=config).draw(
            temp_max=e_max, temp_min=e_min, show=show, save=False
        )


if __name__ == "__main__":
    main(show=True)
