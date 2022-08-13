"""
Created at 21.08.2019
"""
from argparse import ArgumentParser
from dataclasses import dataclass

import numpy as np

from conmech.helpers.config import Config
from conmech.plotting.drawer import Drawer
from conmech.scenarios.problems import TemperatureDynamic
from conmech.simulations.problem_solver import TemperatureTimeDependent as TDynamicProblemSolver
from examples.p_slope_contact_law import make_slope_contact_law


class TPSlopeContactLaw(make_slope_contact_law(slope=1e1)):
    # @staticmethod  # TODO # 48
    # def g(t):
    #     return 10.7 + t * 0.02
    # return 0.5 + t * 0.01

    @staticmethod
    def h_nu(uN, t):
        g_t = 10.7 + t * 0.02
        if uN > g_t:
            return 100.0 * (uN - g_t)
        return 0

    @staticmethod
    def h_tau(uN, t):
        g_t = 10.7 + t * 0.02
        if uN > g_t:
            return 10.0 * (uN - g_t)
        return 0

    # def jT(self, vTx, vTy):  # TODO # 48
    #     # return np.log(np.linalg.norm(vTx, vTy)+1)
    #     return np.linalg.norm(vTx, vTy)

    @staticmethod
    def h_temp(u_tau):  # potential  # TODO # 48
        return 0.1 * 0.5 * u_tau**2


@dataclass()
class TDynamicSetup(TemperatureDynamic):
    grid_height: ... = 1.0
    elements_number: ... = (4, 10)
    mu_coef: ... = 4
    la_coef: ... = 4
    th_coef: ... = 4
    ze_coef: ... = 4
    time_step: ... = 0.02
    contact_law: ... = TPSlopeContactLaw
    thermal_expansion: ... = np.array([[0.5, 0.0, 0.0], [0.0, 0.5, 0.0], [0.0, 0.0, 0.5]])
    thermal_conductivity: ... = np.array([[0.1, 0.0, 0.0], [0.0, 0.1, 0.0], [0.0, 0.0, 0.1]])

    @staticmethod
    def initial_temperature(x: np.ndarray) -> np.ndarray:
        return np.asarray([0.25])

    @staticmethod
    def inner_forces(x):
        return np.array([0.0, -1.0])

    @staticmethod
    def outer_forces(x):
        if x[0] == 0:
            return np.array([48.0 * (0.25 - (x[1] - 0.5) ** 2), 0])
        if x[0] == 2.5:
            return np.array([-48.0 * (0.25 - (x[1] - 0.5) ** 2), 0])
        return np.array([0, 0])

    @staticmethod
    def friction_bound(u_nu):
        return 0

    @staticmethod  # TODO #49
    def is_contact(x):
        return x[1] == 0

    @staticmethod  # TODO #49
    def is_dirichlet(x):
        return False


def main(show: bool = True, save: bool = False):
    setup = TDynamicSetup(mesh_type="cross")
    runner = TDynamicProblemSolver(setup, solving_method="schur")

    states = runner.solve(
        n_steps=32,
        output_step=range(0, 32, 4),
        verbose=True,
        initial_displacement=setup.initial_displacement,
        initial_velocity=setup.initial_velocity,
        initial_temperature=setup.initial_temperature,
    )
    T_max = -np.inf
    T_min = np.inf
    for state in states:
        T_max = max(T_max, np.max(state.temperature))
        T_min = min(T_min, np.min(state.temperature))
    config = Config()
    for state in states:
        Drawer(state=state, config=config).draw(
            temp_max=T_max, temp_min=T_min, show=show, save=save
        )


if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument(
        "--mode",
        type=str,
        choices=["show", "save"],
        default="show",
    )
    args = parser.parse_args()
    save = args.mode == "save"
    main(show=not save, save=save)
