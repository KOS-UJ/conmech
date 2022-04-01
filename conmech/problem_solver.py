"""
General solver for Contact Mechanics problem.
"""
from typing import Callable, List, Optional, Tuple

import numpy as np

from conmech.dataclass.body_properties import (
    DynamicTemperatureBodyProperties, StaticTemperatureBodyProperties)
from conmech.dataclass.mesh_data import MeshData
from conmech.dataclass.schedule import Schedule
from conmech.features.mesh_features import MeshFeatures
from conmech.problems import Dynamic as DynamicProblem
from conmech.problems import Problem
from conmech.problems import Quasistatic as QuasistaticProblem
from conmech.problems import Static as StaticProblem
from conmech.solvers import Solvers
from conmech.solvers.solver import Solver
from conmech.solvers.validator import Validator
from conmech.state import State, TemperatureState


class ProblemSolver:

    def __init__(self, setup: Problem, solving_method: str):
        """Solves general Contact Mechanics problem.

        :param setup:
        :param solving_method: 'schur', 'optimization', 'direct'
        """
        self.C_coeff = np.array([[0.5, 0.0, 0.0], [0.0, 0.5, 0.0], [0.0, 0.0, 0.5]])
        self.K_coeff = np.array([[0.1, 0.0, 0.0], [0.0, 0.1, 0.0], [0.0, 0.0, 0.1]])

        with_time = isinstance(setup, (QuasistaticProblem, DynamicProblem))
        body_prop = DynamicTemperatureBodyProperties(
            mass_density=1.0, mu=setup.mu_coef, lambda_=setup.la_coef, theta=setup.th_coef,
            zeta=setup.ze_coef, C_coeff=self.C_coeff, K_coeff=self.K_coeff
        ) if with_time else StaticTemperatureBodyProperties(
            mass_density=1.0, mu=setup.mu_coef, lambda_=setup.la_coef, C_coeff=self.C_coeff,
            K_coeff=self.K_coeff
        )
        time_step = setup.time_step if with_time else 0

        grid_width = (
                             setup.grid_height / setup.elements_number[0]
                     ) * setup.elements_number[1]

        self.mesh = MeshFeatures(
            mesh_data=MeshData(
                mesh_type="cross",
                mesh_density=[setup.elements_number[1], setup.elements_number[0]],
                scale=[float(grid_width), float(setup.grid_height)],
            ),
            body_prop=body_prop,
            schedule=Schedule(time_step=time_step, final_time=0.0),
            normalize_by_rotation=False,
            is_dirichlet=setup.is_dirichlet,
            is_contact=setup.is_contact,
        )
        self.setup = setup

        self.coordinates = (
            "displacement" if isinstance(setup, StaticProblem) else "velocity"
        )
        self.step_solver: Optional[Solver] = None
        self.validator: Optional[Validator] = None
        self.solving_method = solving_method

    @property
    def solving_method(self):
        return str(self.step_solver)

    @solving_method.setter
    def solving_method(self, value):
        solver_class = Solvers.get_by_name(solver_name=value, problem=self.setup)

        # TODO: fixed solvers to avoid: th_coef, ze_coef = mu_coef, la_coef
        if isinstance(self.setup, StaticProblem):
            time_step = 0
            body_prop = StaticTemperatureBodyProperties(
                mu=self.setup.mu_coef, lambda_=self.setup.la_coef, mass_density=1.0,
                C_coeff=self.C_coeff, K_coeff=self.K_coeff
            )
        elif isinstance(self.setup, (QuasistaticProblem, DynamicProblem)):
            body_prop = DynamicTemperatureBodyProperties(
                mu=self.setup.mu_coef,
                lambda_=self.setup.la_coef,
                theta=self.setup.th_coef,
                zeta=self.setup.ze_coef,
                mass_density=1.0,
                C_coeff=self.C_coeff,
                K_coeff=self.K_coeff
            )
            time_step = self.setup.time_step
        else:
            raise ValueError(f"Unknown problem class: {self.setup.__class__}")

        self.step_solver = solver_class(
            self.mesh,
            self.setup.inner_forces,
            self.setup.outer_forces,
            body_prop,
            time_step,
            self.setup.contact_law,
            self.setup.friction_bound,
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

            solution = self.find_solution(
                self.step_solver,
                state,
                solution,
                self.validator,
                verbose=verbose,
                **kwargs,
            )

            if self.coordinates == "displacement":
                state.set_displacement(solution, t=self.step_solver.currentTime)
                self.step_solver.u_vector[:] = state.displacement.reshape(-1)
            elif self.coordinates == "velocity":
                state.set_velocity(
                    solution, update_displacement=True, t=self.step_solver.currentTime
                )
            else:
                raise ValueError(f"Unknown coordinates: {self.coordinates}")

    def find_solution(
            self, solver, state, solution, validator, *, verbose=False, **kwargs
    ) -> np.ndarray:  # TODO
        quality = 0
        # solution = state[self.coordinates].reshape(2, -1)  # TODO #23
        solution = solver.solve(solution, **kwargs)
        solver.iterate(solution)
        quality = validator.check_quality(state, solution, quality)
        self.print_iteration_info(quality, validator.error_tolerance, verbose)
        return solution

    def find_solution_uzawa(
            self, solver, state, solution, solution_t, *, verbose=False
    ) -> Tuple[np.ndarray, np.ndarray]:
        norm = np.inf
        old_solution = solution.copy().reshape(-1, 1).squeeze()
        old_solution_t = solution_t.copy()
        fuse = 5
        while norm > 1e-3 and bool(fuse):
            fuse -= 1
            solution = self.find_solution(
                solver,
                state,
                solution,
                self.validator,
                temperature=solution_t,
                verbose=verbose,
            )
            solution_t = solver.solve_t(solution_t, solution)
            solver.t_vector = solution_t
            norm = (
                           np.linalg.norm(solution - old_solution) ** 2
                           + np.linalg.norm(old_solution_t - solution_t) ** 2
                   ) ** 0.5
            old_solution = solution.copy()
            old_solution_t = solution_t.copy()
        return solution, solution_t

    @staticmethod
    def print_iteration_info(quality: float, error_tolerance: float, verbose: bool):
        qualitative = quality > error_tolerance
        sign = ">" if qualitative else "<"
        end = "." if qualitative else ", trying again..."
        if verbose:
            print(f"quality = {quality} {sign} {error_tolerance}{end}")


class Static(ProblemSolver):
    def __init__(self, setup, solving_method: str):
        """Solves general Contact Mechanics problem.

        :param setup:
        :param solving_method: 'schur', 'optimization', 'direct'
        """
        super().__init__(setup, solving_method)

        self.coordinates = "displacement"
        self.solving_method = solving_method

    def solve(
            self, initial_displacement: Callable, verbose: bool = False, **kwargs
    ) -> State:
        """
        :param initial_displacement: for the solver
        :param verbose: show prints
        :return: state
        """
        state = State(self.mesh)
        state.displacement = initial_displacement(
            self.mesh.initial_nodes[:self.mesh.independent_nodes_count])

        solution = state.displacement.reshape(2, -1)

        self.step_solver.u_vector[:] = state.displacement.ravel().copy()

        self.run(solution, state, n_steps=1, verbose=verbose, **kwargs)

        return state


class Quasistatic(ProblemSolver):
    def __init__(self, setup, solving_method: str):
        """Solves general Contact Mechanics problem.

        :param setup:
        :param solving_method: 'schur', 'optimization', 'direct'
        """
        super().__init__(setup, solving_method)

        self.coordinates = "velocity"
        self.solving_method = solving_method

    def solve(
            self,
            n_steps: int,
            initial_displacement: Callable,
            initial_velocity: Callable,
            output_step: Optional[iter] = None,
            verbose: bool = False,
    ) -> List[State]:
        """
        :param n_steps: number of time-step in simulation
        :param output_step: from which time-step we want to get copy of State,
                            default (n_steps-1,)
                            example: for Setup.time-step = 2, n_steps = 10,  output_step = (2, 6, 9)
                                     we get 3 shared copy of State for time-steps 4, 12 and 18
        :param initial_displacement: for the solver
        :param initial_velocity: for the solver
        :param verbose: show prints
        :return: state
        """
        output_step = (0, *output_step) if output_step else (0, n_steps)  # 0 for diff

        state = State(self.mesh)
        state.displacement[:] = initial_displacement(
            self.mesh.initial_nodes[:self.mesh.independent_nodes_count])
        state.velocity[:] = initial_displacement(
            self.mesh.initial_nodes[:self.mesh.independent_nodes_count])

        solution = state.velocity.reshape(2, -1)

        self.step_solver.u_vector[:] = state.displacement.ravel().copy()
        self.step_solver.v_vector[:] = state.velocity.ravel().copy()

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

        self.coordinates = "velocity"
        self.solving_method = solving_method

    def solve(
            self,
            n_steps: int,
            initial_displacement: Callable,
            initial_velocity: Callable,
            output_step: Optional[iter] = None,
            verbose: bool = False,
    ) -> List[State]:
        """
        :param n_steps: number of time-step in simulation
        :param output_step: from which time-step we want to get copy of State,
                            default (n_steps-1,)
                            example: for Setup.time-step = 2, n_steps = 10,  output_step = (2, 6, 9)
                                     we get 3 shared copy of State for time-steps 4, 12 and 18
        :param initial_displacement: for the solver
        :param initial_velocity: for the solver
        :param verbose: show prints
        :return: state
        """
        output_step = (0, *output_step) if output_step else (0, n_steps)  # 0 for diff

        state = State(self.mesh)
        state.displacement[:] = initial_displacement(
            self.mesh.initial_nodes[:self.mesh.independent_nodes_count])
        state.velocity[:] = initial_displacement(
            self.mesh.initial_nodes[:self.mesh.independent_nodes_count])

        solution = state.velocity.reshape(2, -1)

        self.step_solver.u_vector[:] = state.displacement.ravel().copy()
        self.step_solver.v_vector[:] = state.velocity.ravel().copy()

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

        self.coordinates = "velocity"
        self.solving_method = solving_method

    def solve(
            self,
            n_steps: int,
            initial_displacement: Callable,
            initial_velocity: Callable,
            initial_temperature: Callable,
            output_step: Optional[iter] = None,
            verbose: bool = False,
    ) -> List[TemperatureState]:
        """
        :param n_steps: number of time-step in simulation
        :param output_step: from which time-step we want to get copy of State,
                            default (n_steps-1,)
                            example: for Setup.time-step = 2, n_steps = 10,  output_step = (2, 6, 9)
                                     we get 3 shared copy of State for time-steps 4, 12 and 18
        :param initial_displacement: for the solver
        :param initial_velocity: for the solver
        :param initial_temperature: for the solver
        :param verbose: show prints
        :return: state
        """
        output_step = (0, *output_step) if output_step else (0, n_steps)  # 0 for diff

        state = TemperatureState(self.mesh)
        state.displacement[:] = initial_displacement(
            self.mesh.initial_nodes[:self.mesh.independent_nodes_count])
        state.velocity[:] = initial_displacement(
            self.mesh.initial_nodes[:self.mesh.independent_nodes_count])
        state.temperature[:] = initial_temperature(
            self.mesh.initial_nodes[:self.mesh.independent_nodes_count])

        solution = state.velocity.reshape(2, -1)
        solution_t = state.temperature

        self.step_solver.u_vector[:] = state.displacement.ravel().copy()
        self.step_solver.v_vector[:] = state.velocity.ravel().copy()
        self.step_solver.t_vector[:] = state.temperature.ravel().copy()

        output_step = np.diff(output_step)
        results = []
        for n in output_step:
            for i in range(n):
                self.step_solver.currentTime += self.step_solver.time_step

                # solution = self.find_solution(self.step_solver, state, solution, self.validator,
                #                               verbose=verbose)
                solution, solution_t = self.find_solution_uzawa(
                    self.step_solver, state, solution, solution_t, verbose=verbose
                )

                if self.coordinates == "velocity":
                    state.set_velocity(
                        solution[:],
                        update_displacement=True,
                        t=self.step_solver.currentTime,
                    )
                    state.set_temperature(solution_t)
                    # self.step_solver.iterate(solution)
                else:
                    raise ValueError(f"Unknown coordinates: {self.coordinates}")
            results.append(state.copy())

        return results
