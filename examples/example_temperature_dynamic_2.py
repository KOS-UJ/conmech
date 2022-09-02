"""
Created at 21.08.2019
"""
from dataclasses import dataclass

import numpy as np
import pickle


from conmech.helpers.config import Config
from conmech.mesh.boundaries_description import BoundariesDescription
from conmech.plotting.drawer import Drawer
from conmech.scenarios.problems import TemperatureDynamic
from conmech.simulations.problem_solver import TemperatureTimeDependent as TDynamicProblemSolver
from examples.p_slope_contact_law import make_slope_contact_law


class TPSlopeContactLaw(make_slope_contact_law(slope=1e1)):

    @staticmethod
    def potential_normal_direction(u_nu: float) -> float:
        if u_nu <= 0:
            return 0.0
        return (0.5 * 1e3 * u_nu) * u_nu

    @staticmethod
    def subderivative_normal_direction(u_nu: float, v_nu: float) -> float:
        if u_nu <= 0:
            return 0 * v_nu
        return (1e3 * u_nu) * v_nu

    @staticmethod
    def potential_tangential_direction(u_tau: np.ndarray) -> float:
        return np.log(np.sum(u_tau * u_tau) ** 0.5 + 1)

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

    @staticmethod
    def temp_exchange(temp):  # potential  # TODO # 48
        return 0.1 * 0.5 * (temp - 0.27) ** 2

    @staticmethod
    def h_temp(u_tau):  # potential  # TODO # 48
        return 0.1 * 0.5 * u_tau**2


@dataclass()
class TDynamicSetup(TemperatureDynamic):
    grid_height: ... = 1.0
    elements_number: ... = (4, 4)
    mu_coef: ... = 45
    la_coef: ... = 105
    th_coef: ... = 4.5
    ze_coef: ... = 10.5
    time_step: ... = 0.01
    contact_law: ... = TPSlopeContactLaw
    thermal_expansion: ... = np.eye(3) * 0.5

    thermal_conductivity: ... = np.eye(3) * 0.033

    @staticmethod
    def initial_temperature(x: np.ndarray) -> np.ndarray:
        return np.asarray([0.25])

    @staticmethod
    def inner_forces(x):
        return np.array([0.0, -9.81e-1])

    @staticmethod
    def outer_forces(x):
        if x[0] == 0:
            return np.array([48.0 * (0.25 - (x[1] - 0.5) ** 2), 0])
        if x[0] == 1.5:
            return np.array([-44.0 * (0.25 - (x[1] - 0.5) ** 2), 0])
        return np.array([0, 0])

    @staticmethod
    def friction_bound(u_nu):
        return 1

    boundaries: ... = BoundariesDescription(contact=lambda x: x[1] == 0)


def main(steps, setup, show: bool = True, save: bool = False):
    setup = setup or TDynamicSetup(mesh_type="cross")
    runner = TDynamicProblemSolver(setup, solving_method="schur")

    states = runner.solve(
        n_steps=steps,
        output_step=(steps,),
        verbose=True,
        initial_displacement=setup.initial_displacement,
        initial_velocity=setup.initial_velocity,
        initial_temperature=setup.initial_temperature,
    )
    config = Config()
    with open(f'output/temp/k_{int(np.log2(steps))}_h_{int(np.log2(setup.elements_number[0]))}',
              'wb') as output:
        pickle.dump(states[-1], output)

    with open(f'output/temp/k_{int(np.log2(steps))}_h_{int(np.log2(setup.elements_number[0]))}',
              'rb') as output:
        state = pickle.load(output)
        print(f"k_{int(np.log2(steps))}_h_{int(np.log2(setup.elements_number[0]))}")
        # Drawer(state=state, config=config).draw(
        #     temp_max=np.max(state.temperature), temp_min=np.min(state.temperature), show=show, save=save
        # )


if __name__ == "__main__":
    T = 1
    ks = [2**i for i in range(0, 8)]
    hs = [2**i for i in range(0, 4)]
    for h in hs:
        for k in ks:
            setup = TDynamicSetup(mesh_type="cross")
            setup.elements_number = (h, 1.5 * h)
            setup.time_step = T / k
            main(setup=setup, steps=k, show=True, save=False)
