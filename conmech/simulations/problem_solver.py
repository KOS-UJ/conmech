"""
General solver for Contact Mechanics problem.
"""
from typing import Callable, List, Optional, Tuple, Type

import numpy as np

from conmech.dynamics.dynamics import DynamicsConfiguration
from conmech.dynamics.statement import (
    StaticDisplacementStatement,
    QuasistaticVelocityStatement,
    DynamicVelocityWithTemperatureStatement,
    TemperatureStatement,
    PiezoelectricStatement,
    DynamicVelocityStatement,
    QuasistaticVelocityWithPiezoelectricStatement, StaticPoissonStatement,
)
from conmech.properties.body_properties import (
    TimeDependentTemperatureBodyProperties,
    BodyProperties,
    TimeDependentBodyProperties,
    StaticBodyProperties,
    TimeDependentPiezoelectricBodyProperties,
)
from conmech.properties.mesh_properties import MeshProperties
from conmech.properties.schedule import Schedule
from conmech.scenarios.problems import (
    TimeDependentProblem,
    Problem,
    StaticProblem,
    QuasistaticProblem,
    PoissonProblem,
    DisplacementProblem,
    StaticDisplacementProblem,
    TimeDependentDisplacementProblem,
    TemperatureTimeDependentProblem,
    TemperatureDynamicProblem,
    PiezoelectricTimeDependentProblem,
    PiezoelectricQuasistaticProblem, ContactLaw,
)
from conmech.scene.body_forces import BodyForces
from conmech.solvers._solvers import SolversRegistry
from conmech.solvers.solver import Solver
from conmech.solvers.validator import Validator
from conmech.state.state import State, TemperatureState, PiezoelectricState


class ProblemSolver:
    def __init__(self, problem: Problem, body_properties: BodyProperties):
        """Solves general Contact Mechanics problem.

        :param problem:
        :param body_properties:
        """
        self.time_step: float
        if isinstance(problem, TimeDependentProblem):
            self.time_step = problem.time_step
        else:
            self.time_step = 0

        grid_width: float = (problem.grid_height / problem.elements_number[0]) * problem.elements_number[1]

        self.body: BodyForces = BodyForces(
            mesh_prop=MeshProperties(
                dimension=2,
                mesh_type=problem.mesh_type,
                mesh_density=[problem.elements_number[1], problem.elements_number[0]],
                scale=[float(grid_width), float(problem.grid_height)],
            ),
            body_prop=body_properties,
            schedule=Schedule(time_step=self.time_step, final_time=0.0),
            boundaries_description=problem.boundaries,
            dynamics_config=DynamicsConfiguration(
                normalize_by_rotation=False,
                create_in_subprocess=False,
                with_lhs=False,
                with_schur=False,
            ),
        )
        self.body.set_permanent_forces_by_functions(
            inner_forces_function=problem.inner_forces, outer_forces_function=problem.outer_forces
        )
        self.problem: Problem = problem

        self.coordinates: Optional[str] = None
        self.step_solver: Optional[Solver] = None
        self.second_step_solver: Optional[Solver] = None
        self.validator: Optional[Validator] = None

    @property
    def solving_method(self) -> str:
        return str(self.step_solver)

    @solving_method.setter
    def solving_method(self, value: str) -> None:
        solver_class: Type[Solver] = SolversRegistry.get_by_name(solver_name=value, problem=self.problem)
        contact_law: Optional[ContactLaw]
        friction_bound: Optional[Callable[[float], float]]

        # TODO: #65 fixed solvers to avoid: th_coef, ze_coef = mu_coef, la_coef
        if isinstance(self.problem, DisplacementProblem):
            contact_law = self.problem.contact_law
            friction_bound = self.problem.friction_bound
            if isinstance(self.problem, StaticProblem):
                statement = StaticDisplacementStatement(self.body)
            elif isinstance(self.problem, TimeDependentProblem):
                if isinstance(self.problem, PiezoelectricQuasistaticProblem):
                    statement = QuasistaticVelocityWithPiezoelectricStatement(self.body)
                elif isinstance(self.problem, QuasistaticProblem):
                    statement = QuasistaticVelocityStatement(self.body)
                elif isinstance(self.problem, TemperatureDynamicProblem):
                    statement = DynamicVelocityWithTemperatureStatement(self.body)
                else:
                    statement = DynamicVelocityStatement(self.body)
            else:
                raise ValueError(f"Unsupported problem class: {self.problem.__class__}")
        elif isinstance(self.problem, PoissonProblem):
            statement = StaticPoissonStatement(self.body)
            contact_law = None
            friction_bound = None
        else:
            raise ValueError(f"Unsupported problem class: {self.problem.__class__}")
        self.step_solver = solver_class(
            statement,
            self.body,
            self.time_step,
            contact_law,
            friction_bound,
        )
        if isinstance(self.problem, TemperatureTimeDependentProblem):
            self.second_step_solver = solver_class(
                TemperatureStatement(self.body),
                self.body,
                self.time_step,
                self.problem.contact_law,
                self.problem.friction_bound,
            )
        elif isinstance(self.problem, PiezoelectricTimeDependentProblem):
            self.second_step_solver = solver_class(
                PiezoelectricStatement(self.body),
                self.body,
                self.time_step,
                self.problem.contact_law,
                self.problem.friction_bound,
            )
        else:
            self.second_step_solver = None
        self.validator = Validator(self.step_solver)

    def solve(self, **kwargs):
        raise NotImplementedError()

    def run(self, solution: np.ndarray, state: State, n_steps: int, verbose: bool = False, **kwargs):
        """
        :param solution:
        :param state:
        :param n_steps: number of steps
        :param verbose: show prints
        :return: state
        """
        for _ in range(n_steps):
            self.step_solver.current_time += self.step_solver.time_step

            solution = self.find_solution(
                state,
                solution,
                self.validator,
                verbose=verbose,
                velocity=solution,
                **kwargs,
            )

            if self.coordinates == "temperature":
                if isinstance(state, TemperatureState):
                    state.set_temperature(solution)
                else:
                    raise ValueError(f"Wrong coordinates type {self.coordinates} for state class {type(solution)}")
            elif self.coordinates == "displacement":
                state.set_displacement(solution, time=self.step_solver.current_time)
                self.step_solver.u_vector[:] = state.displacement.reshape(-1)
            elif self.coordinates == "velocity":
                state.set_velocity(
                    solution,
                    update_displacement=True,
                    time=self.step_solver.current_time,
                )
            else:
                raise ValueError(f"Unknown coordinates: {self.coordinates}")

    def find_solution(self, state: State, solution: np.ndarray, validator: Validator, *, verbose: bool = False,
                      **kwargs) -> np.ndarray:
        quality = 0
        # solution = state[self.coordinates].reshape(2, -1)  # TODO #23
        solution = self.step_solver.solve(solution, **kwargs)

        self.step_solver.iterate(solution)
        if self.second_step_solver is not None:
            self.second_step_solver.iterate(solution)
        # quality = validator.check_quality(state, solution, quality) # TODO Is validator needed?
        # self.print_iteration_info(quality, validator.error_tolerance, verbose)
        return solution

    def find_solution_uzawa(
            self, state: State, solution: np.ndarray, solution_t: np.ndarray, *, verbose: bool = False
    ) -> Tuple[np.ndarray, np.ndarray]:
        norm = np.inf
        old_solution = solution.copy().reshape(-1, 1).squeeze()
        old_solution_t = solution_t.copy()
        fuse = 5
        while norm > 1e-3 and bool(fuse):
            fuse -= 1
            solution = self.find_solution(
                state,
                solution,
                self.validator,
                temperature=solution_t,
                verbose=verbose,
                velocity=solution,
            )
            solution_t = self.second_step_solver.solve(solution_t, velocity=solution)
            self.step_solver.t_vector = solution_t
            self.second_step_solver.t_vector = solution_t
            norm = (
                           np.linalg.norm(solution - old_solution) ** 2
                           + np.linalg.norm(old_solution_t - solution_t) ** 2
                   ) ** 0.5
            old_solution = solution.copy()
            old_solution_t = solution_t.copy()
        return solution, solution_t

    @staticmethod
    def print_iteration_info(quality: float, error_tolerance: float, verbose: bool) -> None:
        qualitative = quality > error_tolerance
        sign = ">" if qualitative else "<"
        end = "." if qualitative else ", trying again..."
        if verbose:
            print(f"quality = {quality} {sign} {error_tolerance}{end}")


class PoissonSolver(ProblemSolver):
    def __init__(self, problem: Problem, solving_method: str):
        """Solves Poisson problem and saves solution as temperature state.

        :param problem:
        :param solving_method: 'schur', 'optimization', 'direct'
        """
        body_prop = StaticBodyProperties(
            mass_density=1.0,
            mu=0,
            lambda_=0,
        )
        super().__init__(problem, body_prop)

        self.coordinates = "temperature"
        self.solving_method = solving_method

    # super class method takes **kwargs, so signatures are consistent
    # pylint: disable=arguments-differ
    def solve(self, *, verbose: bool = False, **kwargs) -> TemperatureState:
        """
        :param verbose: show prints
        :return: state
        """
        state = TemperatureState(self.body)
        state.temperature[:] = 0

        solution_t = state.temperature

        self.step_solver.u_vector[:] = state.displacement.ravel().copy()

        self.run(solution_t, state, n_steps=1, verbose=verbose, **kwargs)

        return state

    def find_solution(self, state, solution, validator, *, verbose=False, **kwargs) -> np.ndarray:
        solution = self.step_solver.solve(solution, **kwargs)
        return solution


class StaticSolver(ProblemSolver):
    def __init__(self, problem: StaticDisplacementProblem, solving_method: str):
        """Solves general Contact Mechanics problem.

        :param problem:
        :param solving_method: 'schur', 'optimization', 'direct'
        """
        body_prop = StaticBodyProperties(
            mass_density=1.0,
            mu=problem.mu_coef,
            lambda_=problem.la_coef,
        )
        super().__init__(problem, body_prop)

        self.coordinates = "displacement"
        self.solving_method = solving_method

    # super class method takes **kwargs, so signatures are consistent
    # pylint: disable=arguments-differ
    def solve(self, *, initial_displacement: Callable, verbose: bool = False, **kwargs) -> State:
        """
        :param initial_displacement: for the solver
        :param verbose: show prints
        :return: state
        """
        state = State(self.body)
        state.displacement = initial_displacement(
            self.body.mesh.initial_nodes[: self.body.mesh.independent_nodes_count]
        )

        solution = state.displacement.reshape(2, -1)

        self.step_solver.u_vector[:] = state.displacement.ravel().copy()

        self.run(solution, state, n_steps=1, verbose=verbose, **kwargs)

        return state


class TimeDependentSolver(ProblemSolver):
    def __init__(self, problem: TimeDependentDisplacementProblem, solving_method: str):
        """Solves general Contact Mechanics problem.

        :param problem:
        :param solving_method: 'schur', 'optimization', 'direct'
        """
        body_prop = TimeDependentBodyProperties(
            mass_density=1.0,
            mu=problem.mu_coef,
            lambda_=problem.la_coef,
            theta=problem.th_coef,
            zeta=problem.ze_coef,
        )
        super().__init__(problem, body_prop)

        self.coordinates = "velocity"
        self.solving_method = solving_method

    # super class method takes **kwargs, so signatures are consistent
    # pylint: disable=arguments-differ
    def solve(
            self,
            *,
            n_steps: int,
            initial_displacement: Callable,
            initial_velocity: Callable,
            output_step: Optional[iter] = None,
            verbose: bool = False,
            **kwargs,
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

        state: State = State(self.body)
        state.displacement[:] = initial_displacement(
            self.body.mesh.initial_nodes[: self.body.mesh.independent_nodes_count]
        )
        state.velocity[:] = initial_velocity(
            self.body.mesh.initial_nodes[: self.body.mesh.independent_nodes_count]
        )

        solution = state.velocity.reshape(2, -1)

        self.step_solver.u_vector[:] = state.displacement.ravel().copy()
        self.step_solver.v_vector[:] = state.velocity.ravel().copy()

        output_step = np.diff(output_step)
        results = []
        for n in output_step:
            self.run(solution, state, n_steps=n, verbose=verbose)
            results.append(state.copy())

        return results


class TemperatureTimeDependentSolver(ProblemSolver):
    def __init__(self, problem: TemperatureTimeDependentProblem, solving_method: str):
        """Solves general Contact Mechanics problem.

        :param problem:
        :param solving_method: 'schur', 'optimization', 'direct'
        """

        body_prop = TimeDependentTemperatureBodyProperties(
            mass_density=1.0,
            mu=problem.mu_coef,
            lambda_=problem.la_coef,
            theta=problem.th_coef,
            zeta=problem.ze_coef,
            thermal_expansion=problem.thermal_expansion,
            thermal_conductivity=problem.thermal_conductivity,
        )
        super().__init__(problem, body_prop)

        self.coordinates = "velocity"
        self.solving_method = solving_method

    # super class method takes **kwargs, so signatures are consistent
    # pylint: disable=arguments-differ
    def solve(
            self,
            *,
            n_steps: int,
            initial_displacement: Callable,
            initial_velocity: Callable,
            initial_temperature: Callable,
            output_step: Optional[iter] = None,
            verbose: bool = False,
            **kwargs,
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

        state = TemperatureState(self.body)
        state.displacement[:] = initial_displacement(
            self.body.mesh.initial_nodes[: self.body.mesh.independent_nodes_count]
        )
        state.velocity[:] = initial_velocity(
            self.body.mesh.initial_nodes[: self.body.mesh.independent_nodes_count]
        )
        state.temperature[:] = initial_temperature(
            self.body.mesh.initial_nodes[: self.body.mesh.independent_nodes_count]
        )

        solution = state.velocity.reshape(2, -1)
        solution_t = state.temperature

        self.step_solver.u_vector[:] = state.displacement.ravel().copy()
        self.step_solver.v_vector[:] = state.velocity.ravel().copy()
        self.step_solver.t_vector[:] = state.temperature.ravel().copy()

        self.second_step_solver.u_vector[:] = state.displacement.ravel().copy()
        self.second_step_solver.v_vector[:] = state.velocity.ravel().copy()
        self.second_step_solver.t_vector[:] = state.temperature.ravel().copy()

        output_step = np.diff(output_step)
        results = []
        for n in output_step:
            for _ in range(n):
                self.step_solver.current_time += self.step_solver.time_step
                self.second_step_solver.current_time += self.second_step_solver.time_step

                # solution = self.find_solution(self.step_solver, state, solution, self.validator,
                #                               verbose=verbose)
                solution, solution_t = self.find_solution_uzawa(
                    state, solution, solution_t, verbose=verbose
                )

                if self.coordinates == "velocity":
                    state.set_velocity(
                        solution[:],
                        update_displacement=True,
                        time=self.step_solver.current_time,
                    )
                    state.set_temperature(solution_t)
                    # self.step_solver.iterate(solution)
                else:
                    raise ValueError(f"Unknown coordinates: {self.coordinates}")
            results.append(state.copy())

        return results


class PiezoelectricTimeDependentSolver(ProblemSolver):
    def __init__(self, problem: PiezoelectricTimeDependentProblem, solving_method: str):
        """Solves general Contact Mechanics problem.

        :param problem:
        :param solving_method: 'schur', 'optimization', 'direct'
        """
        body_prop = TimeDependentPiezoelectricBodyProperties(
            mass_density=1.0,
            mu=problem.mu_coef,
            lambda_=problem.la_coef,
            theta=problem.th_coef,
            zeta=problem.ze_coef,
            piezoelectricity=problem.piezoelectricity,
            permittivity=problem.permittivity,
        )
        super().__init__(problem, body_prop)

        self.coordinates = "velocity"
        self.solving_method = solving_method

    # super class method takes **kwargs, so signatures are consistent
    # pylint: disable=arguments-differ
    def solve(
            self,
            *,
            n_steps: int,
            initial_displacement: Callable,
            initial_velocity: Callable,
            initial_electric_potential: Callable,
            output_step: Optional[iter] = None,
            verbose: bool = False,
            **kwargs,
    ) -> List[PiezoelectricState]:
        """
        :param n_steps: number of time-step in simulation
        :param output_step: from which time-step we want to get copy of State,
                            default (n_steps-1,)
                            example: for Setup.time-step = 2, n_steps = 10,  output_step = (2, 6, 9)
                                     we get 3 shared copy of State for time-steps 4, 12 and 18
        :param initial_displacement: for the solver
        :param initial_velocity: for the solver
        :param initial_electric_potential: for the solver
        :param verbose: show prints
        :return: state
        """
        output_step = (0, *output_step) if output_step else (0, n_steps)  # 0 for diff

        state = PiezoelectricState(self.body)
        state.displacement[:] = initial_displacement(
            self.body.mesh.initial_nodes[: self.body.mesh.independent_nodes_count]
        )
        state.velocity[:] = initial_velocity(
            self.body.mesh.initial_nodes[: self.body.mesh.independent_nodes_count]
        )
        state.electric_potential[:] = initial_electric_potential(
            self.body.mesh.initial_nodes[: self.body.mesh.independent_nodes_count]
        )

        solution = state.velocity.reshape(2, -1)
        solution_t = state.electric_potential

        self.step_solver.u_vector[:] = state.displacement.ravel().copy()
        self.step_solver.v_vector[:] = state.velocity.ravel().copy()
        self.step_solver.p_vector[:] = state.electric_potential.ravel().copy()

        self.second_step_solver.u_vector[:] = state.displacement.ravel().copy()
        self.second_step_solver.v_vector[:] = state.velocity.ravel().copy()
        self.second_step_solver.p_vector[:] = state.electric_potential.ravel().copy()

        output_step = np.diff(output_step)
        results = []
        for n in output_step:
            for _ in range(n):
                self.step_solver.current_time += self.step_solver.time_step
                self.second_step_solver.current_time += self.second_step_solver.time_step

                # solution = self.find_solution(self.step_solver, state, solution, self.validator,
                #                               verbose=verbose)
                solution, solution_t = self.find_solution_uzawa(
                    state, solution, solution_t, verbose=verbose
                )

                if self.coordinates == "velocity":
                    state.set_velocity(
                        solution[:],
                        update_displacement=True,
                        time=self.step_solver.current_time,
                    )
                    state.set_electric_potential(solution_t)
                    # self.step_solver.iterate(solution)
                else:
                    raise ValueError(f"Unknown coordinates: {self.coordinates}")
            results.append(state.copy())

        return results
