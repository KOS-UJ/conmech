"""
Created at 21.08.2019

@author: Michał Jureczka
@author: Piotr Bartman
"""

import numpy as np
from scipy.optimize import fsolve

from simulation.grid_factory import GridFactory
from utils.drawer import Drawer
from simulation.solver import Solver


class SimulationRunner:
    @staticmethod
    def run(setup):
        grid = GridFactory.construct(setup.gridSizeH, setup.gridSizeL, setup.gridHeight)
        solver = Solver(grid, setup.timeStep, setup.F0, setup.FN, setup.mi, setup.la)
        d = Drawer(solver)

        u_vector = np.zeros([2 * grid.indNumber()])

        # solver.currentTime = 1
        solver.F.setF()

        while True:
            u_vector = fsolve(solver.f, u_vector)
            quality_inv = np.linalg.norm(solver.f(u_vector))
            if quality_inv < 1:
                break
            else:
                print(f"Quality = {quality_inv**-1} is too low, trying again...")

        solver.iterate(u_vector)
        d.draw()
