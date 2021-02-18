"""
Created at 21.08.2019
"""

import numpy as np

from simulation.grid_factory import GridFactory
from simulation.state import State
from simulation.solvers.solver import Solver
from simulation.solvers.direct import Direct
from simulation.solvers.optimization import Optimization
from simulation.solvers.validator import Validator


class SimulationRunner:
    THRESHOLD = 1

    def __init__(self, setup):
        self.grid = GridFactory.construct(setup.cells_number[0],
                                          setup.cells_number[1],
                                          setup.grid_height
                                          )
        self.setup = setup

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
        displacement = SimulationRunner.find_solution(
            solver, state, validator, initial_guess=initial_guess, verbose=verbose)
        state.set_u_and_displaced_points(displacement)
        return state

    @staticmethod
    def find_solution(solver, state, validator, initial_guess, verbose=False) -> np.ndarray:
        quality = 0
        iteration = 0
        displacement = initial_guess or np.zeros(2 * state.grid.indNumber())
        while quality < SimulationRunner.THRESHOLD:
            displacement = solver.solve(displacement)
            quality = validator.validate(state, displacement)
            iteration += 1
            SimulationRunner.print_iteration_info(iteration, quality, verbose)
        return displacement

    @staticmethod
    def get_solver_class(method: str) -> type:
        if method == 'direct':
            solver_class = Direct
        elif method == 'optimization':
            solver_class = Optimization
        else:
            raise ValueError()
        return solver_class

    def get_solver(self, setup, method: str) -> Solver:
        solver_class = self.get_solver_class(method)
        solver = solver_class(self.grid,
                              setup.inner_forces, setup.outer_forces,
                              setup.mu_coef, setup.lambda_coef,
                              setup.ContactLaw,
                              setup.friction_bound
                              )
        return solver

    @staticmethod
    def print_iteration_info(iteration, quality, verbose):
        qualitative = quality > SimulationRunner.THRESHOLD
        sign = ">" if qualitative else "<"
        end = "." if qualitative else ", trying again..."
        if verbose:
            print(f"iteration = {iteration}; quality = {quality} {sign} {SimulationRunner.THRESHOLD}{end}")
