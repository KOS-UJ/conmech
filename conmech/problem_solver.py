"""
General solver for Contact Mechanics problem.
"""
from typing import Optional, List

import numpy as np

from conmech.grid_factory import GridFactory
from conmech.state import State
from conmech.solvers.solver import Solver
from conmech.solvers import get_solver_class
from conmech.solvers.validator import Validator
from conmech.problems import Problem
from conmech.problems import Static as StaticProblem
from conmech.problems import Quasistatic as QuasistaticProblem
from conmech.problems import Dynamic as DynamicProblem
from conmech.solvers.coefficients import Coefficients


class ProblemSolver:

    def __init__(self, setup: Problem, solving_method: str):
        """Solves general Contact Mechanics problem.

        :param setup:
        :param solving_method: 'schur', 'optimization', 'direct'
        """
        self.grid = GridFactory.construct(setup.cells_number[0],
                                          setup.cells_number[1],
                                          setup.grid_height
                                          )
        self.setup = setup

        self.coordinates = 'displacement' if isinstance(setup, StaticProblem) else 'velocity'
        self.step_solver: Optional[Solver] = None
        self.validator: Optional[Validator] = None
        self.solving_method = solving_method

    @property
    def solving_method(self):
        return str(self.step_solver)

    @solving_method.setter
    def solving_method(self, value):
        solver_class = get_solver_class(value, self.setup)

        # TODO: fixed solvers to avoid: th_coef, ze_coef = mu_coef, lambda_coef
        if isinstance(self.setup, StaticProblem):
            time_step = 0
            coefficients = Coefficients(mu=self.setup.mu_coef, lambda_=self.setup.lambda_coef)
        elif isinstance(self.setup, (QuasistaticProblem, DynamicProblem)):
            coefficients = Coefficients(mu=self.setup.mu_coef, lambda_=self.setup.lambda_coef,
                                        theta=self.setup.th_coef, zeta=self.setup.ze_coef)
            time_step = self.setup.time_step
        else:
            raise ValueError(f"Unknown problem class: {self.setup.__class__}")

        self.step_solver = solver_class(self.grid,
                                        self.setup.inner_forces, self.setup.outer_forces,
                                        coefficients,
                                        time_step,
                                        self.setup.contact_law,
                                        self.setup.friction_bound
                                        )
        self.validator = Validator(self.step_solver)

    def solve(self, *args, **kwargs):
        raise NotImplementedError()

    def run(self, state: State, n_steps: int, verbose: bool = False):
        """
        :param state:
        :param n_steps: number of steps
        :param verbose: show prints
        :return: state
        """
        for i in range(n_steps):
            self.step_solver.currentTime += self.step_solver.time_step

            solution = self.find_solution(self.step_solver, state, self.validator, verbose=verbose)

            if self.coordinates == 'displacement':
                state.set_displacement(solution, t=self.step_solver.currentTime)
            elif self.coordinates == 'velocity':
                state.set_velocity(solution,
                                   update_displacement=True,
                                   t=self.step_solver.currentTime)
                #################### ADDED
                self.step_solver.iterate(state.velocity)
                ####################
            else:
                raise ValueError(f"Unknown coordinates: {self.coordinates}")

    def find_solution(self, solver, state, validator, verbose=False) -> np.ndarray:  # TODO
        quality = 0
        iteration = 0
        solution = state[self.coordinates].reshape(2, -1)
        while quality < validator.error_tolerance:
            solution = solver.solve(solution)
            quality = validator.check_quality(state, solution, quality)
            iteration += 1
            self.print_iteration_info(iteration, quality, validator.error_tolerance, verbose)
        return solution

    @staticmethod
    def print_iteration_info(iteration: int, quality: float, error_tolerance: float, verbose: bool):
        qualitative = quality > error_tolerance
        sign = ">" if qualitative else "<"
        end = "." if qualitative else ", trying again..."
        if verbose:
            print(f"iteration = {iteration}; quality = {quality} {sign} {error_tolerance}{end}")


class Static(ProblemSolver):

    def __init__(self, setup, solving_method: str):
        """Solves general Contact Mechanics problem.

        :param setup:
        :param solving_method: 'schur', 'optimization', 'direct'
        """
        super().__init__(setup, solving_method)

        self.coordinates = 'displacement'
        self.solving_method = solving_method

    def solve(self, initial_displacement: np.ndarray = None, verbose: bool = False) -> State:
        """
        :param initial_displacement: for the solver
        :param verbose: show prints
        :return: state
        """
        state = State(self.grid)
        if initial_displacement:
            state.set_displacement(initial_displacement)

        self.run(state, n_steps=1, verbose=verbose)

        return state


class Quasistatic(ProblemSolver):

    def __init__(self, setup, solving_method: str):
        """Solves general Contact Mechanics problem.

        :param setup:
        :param solving_method: 'schur', 'optimization', 'direct'
        """
        super().__init__(setup, solving_method)

        self.coordinates = 'velocity'
        self.solving_method = solving_method

    def solve(self, n_steps: int, output_step: Optional[iter] = None,
              initial_velocity: np.ndarray = None, verbose: bool = False) -> List[State]:
        """
        :param n_steps: number of time-step in simulation
        :param output_step: from which time-step we want to get copy of State,
                            default (n_steps-1,)
                            example: for Setup.time-step = 2, n_steps = 10,  output_step = (2, 6, 9)
                                     we get 3 shared copy of State for time-steps 4, 12 and 18
        :param initial_velocity: for the solver
        :param verbose: show prints
        :return: state
        """
        output_step = (0, *output_step) if output_step else (0, n_steps)  # 0 for diff

        state = State(self.grid)
        if initial_velocity:
            state.set_velocity(initial_velocity, update_displacement=False)

        output_step = np.diff(output_step)
        results = []
        for n in output_step:
            self.run(state, n_steps=n, verbose=verbose)
            results.append(state.copy())

        return results

class Dynamic(ProblemSolver):

    def __init__(self, setup, solving_method: str):
        """Solves general Contact Mechanics problem.

        :param setup:
        :param solving_method: 'schur', 'optimization', 'direct'
        """
        super().__init__(setup, solving_method)

        self.coordinates = 'velocity'
        self.solving_method = solving_method

    def solve(self, n_steps: int, output_step: Optional[iter] = None,
              initial_velocity: np.ndarray = None, verbose: bool = False) -> List[State]:
        """
        :param n_steps: number of time-step in simulation
        :param output_step: from which time-step we want to get copy of State,
                            default (n_steps-1,)
                            example: for Setup.time-step = 2, n_steps = 10,  output_step = (2, 6, 9)
                                     we get 3 shared copy of State for time-steps 4, 12 and 18
        :param initial_velocity: for the solver
        :param verbose: show prints
        :return: state
        """
        output_step = (0, *output_step) if output_step else (0, n_steps)  # 0 for diff

        state = State(self.grid)
        if initial_velocity:
            state.set_velocity(initial_velocity, update_displacement=False)

        output_step = np.diff(output_step)
        results = []
        for n in output_step:
            self.run(state, n_steps=n, verbose=verbose)
            results.append(state.copy())

        return results
