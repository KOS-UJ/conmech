"""
Created at 21.08.2019
"""

import numpy as np
from simulation.simulation_runner import SimulationRunner
from utils.drawer import Drawer


p_slope = 1.


class Setup:
    time_step = 1
    grid_height = 1

    cells_number = (2, 5)  # number of triangles per aside
    inner_forces = np.array([-0.2, -0.2])
    outer_forces = np.array([0, 0])
    mu_coef = 4
    lambda_coef = 4

    class ContactLaw:
        @staticmethod
        def potential_normal_direction(u_nu: float) -> float:
            if u_nu <= 0:
                return 0.
            return (0.5 * p_slope * u_nu) * u_nu

        @staticmethod
        def subderivative_normal_direction(u_nu: float, v_nu: float) -> float:
            if u_nu <= 0:
                return 0 * v_nu
            return (p_slope * u_nu) * v_nu

        @staticmethod
        def regularized_subderivative_tangential_direction(u_tau: np.ndarray, v_tau: np.ndarray, rho=1e-7) -> float:
            """
            Coulomb regularization
            """
            regularization = 1 / np.sqrt(u_tau[0] * u_tau[0] + u_tau[1] * u_tau[1] + rho ** 2)
            result = regularization * (u_tau[0] * v_tau[0] + u_tau[1] * v_tau[1])
            return result

    @staticmethod
    def friction_bound(u_nu):
        return 0


if __name__ == '__main__':
    setup = Setup()
    runner = SimulationRunner(setup)

    state = runner.run(method='direct', verbose=True)
    Drawer(state).draw()

    state = runner.run(method='optimization', verbose=True)
    Drawer(state).draw()
