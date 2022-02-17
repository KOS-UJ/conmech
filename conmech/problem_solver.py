"""
General solver for Contact Mechanics problem.
"""
from typing import Optional, List, Tuple

import numpy as np

from conmech.grid_factory import GridFactory
from conmech.state import State, TemperatureState
from conmech.solvers.solver import Solver
from conmech.solvers import Solvers
from conmech.solvers.validator import Validator
from conmech.problems import Problem
from conmech.problems import Static as StaticProblem
from conmech.problems import Quasistatic as QuasistaticProblem
from conmech.problems import Dynamic as DynamicProblem
from conmech.solvers.coefficients import Coefficients
from graph.mesh_features import MeshFeatures


class ProblemSolver:

    def __init__(self, setup: Problem, solving_method: str):
        """Solves general Contact Mechanics problem.

        :param setup:
        :param solving_method: 'schur', 'optimization', 'direct'
        """
        # TODO, second size
        # self.grid = GridFactory.construct(setup.cells_number[0],
        #                                   setup.cells_number[1],
        #                                   setup.grid_height
        #                                   )

        th_coef = setup.th_coef if hasattr(setup, "th_coef") else 0
        ze_coef = setup.ze_coef if hasattr(setup, "ze_coef") else 0
        time_step = setup.time_step if hasattr(setup, "time_step") else 0
        dens = 1

        self.grid = MeshFeatures(setup.cells_number[0], "cross",
                                 corners=[0., 0., setup.grid_height, setup.grid_height],
                                 is_adaptive=False,
                                 MU=setup.mu_coef, LA=setup.lambda_coef, TH=th_coef, ZE=ze_coef,
                                 DENS=dens, TIMESTEP=time_step,
                                 is_dirichlet=setup.is_dirichlet,
                                 is_contact=setup.is_contact)
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
        solver_class = Solvers.get_by_name(solver_name=value, problem=self.setup)

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

    def run(self, solution, state, n_steps: int, verbose: bool = False, **kwargs):
        """
        :param state:
        :param n_steps: number of steps
        :param verbose: show prints
        :return: state
        """
        for i in range(n_steps):
            self.step_solver.currentTime += self.step_solver.time_step

            solution = self.find_solution(self.step_solver, state, solution, self.validator, verbose=verbose, **kwargs)

            if self.coordinates == 'displacement':
                state.set_displacement(solution, t=self.step_solver.currentTime)
                self.step_solver.u_vector = state.displacement.reshape(-1)
            elif self.coordinates == 'velocity':
                state.set_velocity(solution,
                                   update_displacement=True,
                                   t=self.step_solver.currentTime)
                #################### ADDED
                # self.step_solver.iterate(solution)
                ####################
            else:
                raise ValueError(f"Unknown coordinates: {self.coordinates}")

    def find_solution(
            self, solver, state, solution, validator, *, verbose=False, **kwargs) -> np.ndarray:  # TODO
        quality = 0
        iteration = 0
        fuse = 1
        # solution = state[self.coordinates].reshape(2, -1)  # TODO #23
        while quality < validator.error_tolerance and bool(fuse):
            fuse -= 1
            solution = solver.solve(solution, **kwargs)
            quality = 0  # TODO validator.check_quality(state, solution, quality)
            iteration += 1
            self.print_iteration_info(iteration, quality, validator.error_tolerance, verbose)
        return solution

    def find_solution_uzawa(self, solver, state, solution, solution_t, *, verbose=False) -> Tuple[np.ndarray, np.ndarray]:
        norm = np.inf
        old_solution = solution.copy().reshape(-1, 1).squeeze()
        old_solution_t = solution_t.copy()
        fuse = 5
        while norm > 1e-3 and bool(fuse):
            fuse -= 1
            solution = self.find_solution(solver, state, solution, self.validator, temperature=solution_t, verbose=verbose)
            solution_t = solver.solve_t(solution_t, solution)
            norm = (np.linalg.norm(solution - old_solution)**2
                    + np.linalg.norm(old_solution_t - solution_t)**2)**0.5
            old_solution = solution.copy()
            old_solution_t = solution_t.copy()
        return solution, solution_t

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

    def solve(
            self,
            initial_displacement: np.ndarray = None,
            verbose: bool = False,
            **kwargs
    ) -> State:
        """
        :param initial_displacement: for the solver
        :param verbose: show prints
        :return: state
        """
        state = State(self.grid)
        if initial_displacement:
            state.set_displacement(initial_displacement)
        solution = state.displacement.reshape(2, -1)

        self.run(solution, state, n_steps=1, verbose=verbose, **kwargs)

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
        solution = state.velocity.reshape(2, -1)

        output_step = np.diff(output_step)
        results = []
        for n in output_step:
            self.run(solution, state, n_steps=n, verbose=verbose)
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
        solution = state.velocity.reshape(2, -1)

        output_step = np.diff(output_step)
        results = []
        for n in output_step:
            self.run(solution, state, n_steps=n, verbose=verbose)
            results.append(state.copy())

        return results


class TDynamic(ProblemSolver):

    def __init__(self, setup, solving_method: str):
        """Solves general Contact Mechanics problem.

        :param setup:
        :param solving_method: 'schur', 'optimization', 'direct'
        """
        super().__init__(setup, solving_method)

        self.coordinates = 'velocity'
        self.solving_method = solving_method

    def solve(self, n_steps: int, output_step: Optional[iter] = None,
              initial_velocity: np.ndarray = None, verbose: bool = False) -> List[TemperatureState]:
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

        state = TemperatureState(self.grid)
        if initial_velocity:
            state.set_velocity(initial_velocity, update_displacement=False)
        solution = state.velocity.reshape(2, -1)
        solution_t = state.temperature

        output_step = np.diff(output_step)
        results = []
        for n in output_step:
            for i in range(n):
                self.step_solver.currentTime += self.step_solver.time_step

                # solution = self.find_solution(self.step_solver, state, solution, self.validator,
                #                               verbose=verbose)
                solution, solution_t = self.find_solution_uzawa(
                    self.step_solver, state, solution, solution_t, verbose=verbose)

                if self.coordinates == 'velocity':
                    state.set_velocity(solution[:],
                                       update_displacement=True,
                                       t=self.step_solver.currentTime)
                    state.set_temperature(solution_t)
                else:
                    raise ValueError(f"Unknown coordinates: {self.coordinates}")
            results.append(state.copy())

        return results
