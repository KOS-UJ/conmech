"""
Created at 21.08.2019
"""

import numpy as np
import scipy.optimize

from simulation.grid_factory import GridFactory
from simulation.solver import Solver, f


class SimulationRunner:

    def __init__(self, setup):
        self.grid = GridFactory.construct(setup.cells_number[0],
                                          setup.cells_number[1],
                                          setup.gridHeight)
        self.solver = Solver(self.grid, setup.F0, setup.FN, setup.mi, setup.la)

        self.solver.F.setF()

    def run(self, start_u=None):
        grid = self.grid
        solver = self.solver
        u_vector = start_u or np.zeros(2 * grid.indNumber())

        quality = 0
        iteration = 0
        while quality < 100:
            if iteration > 0:
                print(f"iteration = {iteration}; quality = {quality} is too low, trying again...")
            u_vector = scipy.optimize.fsolve(
                f, u_vector,
                args=(grid.indNumber(), grid.BorderEdgesD, grid.BorderEdgesN, grid.BorderEdgesC, grid.Edges,
                      grid.Points, solver.knu, solver.B, solver.F.Zero, solver.F.One))
            quality_inv = np.linalg.norm(
                f(u_vector, grid.indNumber(), grid.BorderEdgesD, grid.BorderEdgesN, grid.BorderEdgesC, grid.Edges,
                  grid.Points, solver.knu, solver.B, solver.F.Zero, solver.F.One))
            quality = quality_inv ** -1
            iteration += 1

        print(f"iteration = {iteration}; quality = {quality} is acceptable.")

        solver.set_u_and_displaced_points(u_vector)

        return solver
