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
                             setup.potential_contact_law_normal_direction,
                             setup.contact_law_tangential_direction,
                             setup.friction_bound)

        self.solver.F.setF()

    def run(self, start_u=None, method='optimization'): #method = 'optimization' 'equation'
        grid = self.grid
        solver = self.solver
        u_vector = start_u or np.zeros(2 * grid.indNumber())

        quality = 0
        iteration = 0
        while quality < 8:
            if iteration > 0:
                print(f"iteration = {iteration}; quality = {quality} is too low, trying again...")

            if(method == 'equation'):
                #print("value on equation starting vector for equation", solver.f(u_vector, grid.indNumber(), grid.BorderEdgesD, grid.BorderEdgesN, grid.BorderEdgesC, grid.Edges,
                #          grid.Points, solver.B, solver.F.Zero, solver.F.One))

                u_vector = scipy.optimize.fsolve(
                    solver.f, u_vector,
                    args=(grid.indNumber(), grid.BorderEdgesC, grid.Edges,
                          grid.Points, solver.B, solver.F.Zero, solver.F.One))

                #print("value on equation final vector for equation", solver.f(u_vector, grid.indNumber(), grid.BorderEdgesD, grid.BorderEdgesN, grid.BorderEdgesC, grid.Edges,
                #          grid.Points, solver.B, solver.F.Zero, solver.F.One))

            elif (method == 'optimization'):

                C11 = solver.B[0, 0]
                C12 = solver.B[0, 1]
                C21 = solver.B[1, 0]
                C22 = solver.B[1, 1]
                C = np.bmat([[C11, C12], [C21, C22]])
                E = np.append(solver.F.Zero, solver.F.One)

                #print("value on optimization starting vector for optimization", solver.L2(u_vector, grid.indNumber(), grid.BorderEdgesD, grid.BorderEdgesN,
                #          grid.BorderEdgesC, grid.Edges, grid.Points, C, E))

                u_vector = scipy.optimize.minimize(solver.L2, u_vector,
                    args=(grid.indNumber(), grid.BorderEdgesC, grid.Edges, grid.Points, C, E),
                          method='Powell',
                          options={'disp': True, 'maxiter': len(u_vector) * 1e5},
                          tol=1e-30).x

                #print("value on optimization final vector for optimization", solver.L2(u_vector, grid.indNumber(), grid.BorderEdgesD, grid.BorderEdgesN,
                #          grid.BorderEdgesC, grid.Edges, grid.Points, C, E))

                #print("value on optimization final vector for equation", solver.f(u_vector, grid.indNumber(), grid.BorderEdgesD, grid.BorderEdgesN, grid.BorderEdgesC, grid.Edges,
                #          grid.Points, solver.B, solver.F.Zero, solver.F.One))


            quality_inv = np.linalg.norm(
                solver.f(u_vector, grid.indNumber(), grid.BorderEdgesC, grid.Edges,
                  grid.Points, solver.B, solver.F.Zero, solver.F.One))
            quality = quality_inv ** -1
            iteration += 1

        print(f"iteration = {iteration}; quality = {quality} is acceptable.")

        solver.set_u_and_displaced_points(u_vector)

        return solver
