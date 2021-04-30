"""
Created at 21.08.2019
"""

import numpy as np

from simulation.grid_factory import GridFactory
from simulation.state import State
from simulation.solvers.solver import Solver
from simulation.solvers import get_solver_class
from simulation.solvers.validator import Validator
from utils.drawer import Drawer


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

        velocity = np.zeros(2 * state.grid.independent_num)
        for i in range(1, 10):
            solver.currentTime = i * solver.time_step
            velocity = self.find_solution(
                solver, state, validator, initial_guess=velocity, verbose=verbose)
            solver.iterate(velocity)
            state.set_u_and_displaced_points(solver.u_vector)
            Drawer(state).draw()

        state.set_u_and_displaced_points(solver.u_vector)
        return state

    def find_solution(self, solver, state, validator, initial_guess, verbose=False) -> np.ndarray:
        quality = 0
        iteration = 0
        displacement_or_velocity = initial_guess #or np.zeros(2 * state.grid.independent_num)
        while quality < self.THRESHOLD:
            displacement_or_velocity = solver.solve(displacement_or_velocity)
            quality = validator.check_quality(state, displacement_or_velocity, quality)
            iteration += 1
            self.print_iteration_info(iteration, quality, verbose)
        return displacement_or_velocity

    def get_solver(self, setup, method: str) -> Solver:
        solver_class = get_solver_class(method)
        solver = solver_class(self.grid,
                              setup.inner_forces, setup.outer_forces,
                              setup.mu_coef, setup.lambda_coef,
                              setup.th_coef, setup.ze_coef,
                              setup.time_step,
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
