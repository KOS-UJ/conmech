"""
Created at 21.08.2019

@author: Michał Jureczka
@author: Piotr Bartman
"""

from pylab import *
from scipy.optimize import fsolve

from simulation.grid_factory import GridFactory
from utils.drawer import Drawer
from simulation.solver import Solver


class SimulationRunner:

    def run(self, setup):
        s1 = GridFactory.construct(setup.gridSizeH, setup.gridSizeL, setup.gridHeight)
        solver = Solver(s1, setup.timeStep, setup.F0, setup.FN, setup.mi, setup.la)
        d = Drawer(solver)

        uVector = np.zeros([2 * s1.indNumber()])

        solver.currentTime = 1
        solver.F.setF()

        while True:
            uVector = fsolve(solver.f, uVector)
            quality_inv = np.linalg.norm(solver.f(uVector))
            if quality_inv < 1:
                break
            else:
                print(f"Quality = {quality_inv**-1} is too low, trying again...")

        solver.iterate(uVector)
        d.draw()
