"""
Created at 21.08.2019
"""

import numpy as np

from simulation.grid_factory import GridFactory
from simulation.state import State
from simulation.solvers.solver import Solver
from simulation.solvers import get_solver_class
from simulation.solvers.validator import Validator


class SimulationRunner:

    def __init__(self, setup):
        self.grid = GridFactory.construct(setup.cells_number[0],
                                          setup.cells_number[1],
                                          setup.grid_height
                                          )
        self.setup = setup
        self.THRESHOLD = 1

    def run(self, initial_guess: (np.ndarray, None) = None, method: str = 'direct', verbose: bool = False) -> State:
        """
        :param initial_guess:
        :param method: 'optimization', 'direct'
        :param verbose: show prints
        :return: setup
        """
        setup = self.setup
        solver = self.get_solver(setup, method)
        state = State(self.grid)
        validator = Validator(solver)
        displacement = self.find_solution(
            solver, state, validator, initial_guess=initial_guess, verbose=verbose)
        state.set_u_and_displaced_points(displacement)
        return state

    def find_solution(self, solver, state, validator, initial_guess, verbose=False) -> np.ndarray:
        quality = 0
        iteration = 0
        displacement = initial_guess or np.zeros(2 * state.grid.independent_num())
        while quality < self.THRESHOLD:
            displacement = solver.solve(displacement)
            quality = validator.check_quality(state, displacement, quality)
            iteration += 1
            self.print_iteration_info(iteration, quality, verbose)
        return displacement

    def get_solver(self, setup, method: str) -> Solver:
        solver_class = get_solver_class(method)
        solver = solver_class(self.grid,
                              setup.inner_forces, setup.outer_forces,
                              setup.mu_coef, setup.lambda_coef,
                              setup.ContactLaw,
                              setup.friction_bound
                              )
        return solver

    def print_iteration_info(self, iteration, quality, verbose):
        qualitative = quality > self.THRESHOLD
        sign = ">" if qualitative else "<"
        end = "." if qualitative else ", trying again..."
        if verbose:
            print(f"iteration = {iteration}; quality = {quality} {sign} {self.THRESHOLD}{end}")
