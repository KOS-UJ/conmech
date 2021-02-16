"""
Created at 21.08.2019
"""

import numpy as np
from simulation.simulation_runner import SimulationRunner
from utils.drawer import Drawer


class Setup:
    time_step = 1
    gridHeight = 1

    cells_number = (2, 5)  # number of triangles per aside
    F0 = np.array([-0.2, -0.2])  # inner forces
    FN = np.array([0, 0])  # outer forces
    mi = 4
    la = 4

    @staticmethod
    def contact_law_normal_direction(uN, vN):  # un, vN - scalars
        if uN <= 0:
            return 0 * vN
        return (1. * uN) * vN

    @staticmethod
    def contact_law_tangential_direction(uT, vT, rho=0.0000001):  # uT, vT - vectors; Coulomb regularization
        M = 1 / np.sqrt(uT[0] * uT[0] + uT[1] * uT[1] + rho ** 2)
        result = M * uT[0] * vT[0] + M * uT[1] * vT[1]
        return result

    @staticmethod
    def friction_bound(uN):
        return 0


if __name__ == '__main__':
    setup = Setup()
    runner = SimulationRunner(setup)
    solver = runner.run()
    Drawer(solver).draw()
