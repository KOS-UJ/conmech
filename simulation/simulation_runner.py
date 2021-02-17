"""
Created at 21.08.2019
"""

import numpy as np
import scipy.optimize

from simulation.grid_factory import GridFactory
from simulation.solver import Solver


class SimulationRunner:

    def __init__(self, setup):
        self.grid = GridFactory.construct(setup.cells_number[0],
                                          setup.cells_number[1],
                                          setup.gridHeight)
        self.solver = Solver(self.grid,
                             setup.F0, setup.FN,
                             setup.mi, setup.la,
                             setup.contact_law_normal_direction,
                             setup.contact_law_tangential_direction,
                             setup.friction_bound)

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
                solver.f, u_vector,
                args=(grid.indNumber(), grid.BorderEdgesD, grid.BorderEdgesN, grid.BorderEdgesC, grid.Edges,
                      grid.Points, solver.B, solver.F.Zero, solver.F.One))
            quality_inv = np.linalg.norm(
                solver.f(u_vector, grid.indNumber(), grid.BorderEdgesD, grid.BorderEdgesN, grid.BorderEdgesC, grid.Edges,
                  grid.Points, solver.B, solver.F.Zero, solver.F.One))
            quality = quality_inv ** -1
            iteration += 1

        print(f"iteration = {iteration}; quality = {quality} is acceptable.")

        solver.set_u_and_displaced_points(u_vector)

        return solver
