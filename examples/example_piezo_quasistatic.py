"""
Created at 21.08.2019
"""
from dataclasses import dataclass

import numpy as np

from conmech.helpers.config import Config
from conmech.mesh.boundaries_description import BoundariesDescription
from conmech.simulations.problem_solver import PiezoelectricTimeDependent
from conmech.scenarios.problems import PiezoelectricQuasistatic as PiezoelectricQuasistaticProblem
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
    elements_number: ... = (2, 2)
    mu_coef: ... = 45
    la_coef: ... = 105
    th_coef: ... = 4.5
    ze_coef: ... = 10.5
    time_step: ... = 0.01
    contact_law: ... = PPSlopeContactLaw
    piezoelectricity: ... = np.array([[0.5, 0.0, 0.0], [0.0, 0.5, 0.0], [0.0, 0.0, 0.5]])
    permittivity: ... = np.array([[8.3, 0.0, 0.0], [0.0, 8.8, 0.0], [0.0, 0.0, -8]])

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
        # contact=lambda x: x[1] == 0, dirichlet=lambda x: x[0] == 0,
        # piezo_dirichlet_0=(
        #     lambda x: (x[0] == 1.0 and 0.0 <= x[1] <= 4.0),
        #     lambda x: np.full(x.shape[0], -20)
        # ),
        piezo_dirichlet_0=(
            lambda x: (x[0] == 1.0 and 1.0 <= x[1] <= 4.0),
            lambda x: np.full(x.shape[0], 0)
        ),
        piezo_dirichlet_1=(
            lambda x: (1.5 <= x[0] <= 3.0 and x[1] == 1.0),
            lambda x: np.full(x.shape[0], 20)
        ),
    )


def main(show: bool):
    setup = PQuasistaticSetup(mesh_type="Barboteu2008")
    runner = PiezoelectricTimeDependent(setup, solving_method="global")
    steps = 100
    output = steps // 5
    states = runner.solve(
        n_steps=steps,
        output_step=range(0, steps+1, output),
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
