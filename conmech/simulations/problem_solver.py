"""
General solver for Contact Mechanics problem.
"""

from typing import Callable, List, Optional, Tuple, Type, Iterable

import numpy as np

from conmech.dynamics.dynamics import Dynamics
from conmech.dynamics.statement import Variables
from conmech.mesh.mesh import Mesh
from conmech.properties.body_properties import (
    BodyProperties,
    ElasticProperties,
    ViscoelasticProperties,
    ElasticRelaxationProperties,
    ViscoelasticPiezoelectricProperties,
    ViscoelasticTemperatureProperties,
    MembraneProperties,
)
from conmech.scenarios.problems import (
    TimeDependentProblem,
    Problem,
    PoissonProblem,
    StaticDisplacementProblem,
    TimeDependentDisplacementProblem,
    TemperatureTimeDependentProblem,
    PiezoelectricTimeDependentProblem,
    RelaxationQuasistaticProblem,
    WaveProblem,
)
from conmech.solvers import SchurComplementOptimization
from conmech.solvers._solvers import SolversRegistry
from conmech.solvers.solver import Solver
from conmech.state.products.product import Product
from conmech.state.state import State, TemperatureState, PiezoelectricState


class Body:
    def __init__(self, properties, mesh):
        self.properties: BodyProperties = properties
        self.mesh: Mesh = mesh
        self.dynamics: Optional[Dynamics] = None
        self.state: Optional[State] = None


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

        mesh = Mesh(
            mesh_descr=problem.mesh_descr,
            boundaries_description=problem.boundaries,
        )
        self.body = Body(body_properties, mesh)
        self.schedule = None

        Dynamics(self.body)

        self.body.dynamics.force.inner.source = problem.inner_forces
        self.body.dynamics.force.outer.source = problem.outer_forces
        if isinstance(problem, PoissonProblem):
            self.body.dynamics.temperature.inner.source = problem.internal_temperature
            self.body.dynamics.temperature.outer.source = problem.outer_temperature

        self.problem: Problem = problem

        # True if only one dimension of displacement is used
        self.driving_vector = False

        self.coordinates: Optional[str] = None
        self.step_solver: Optional[Solver] = None
        self.second_step_solver: Optional[Solver] = None

        self.done = 0
        self.to_do = 1

        self.computation_time = None

    @property
    def solving_method(self) -> str:
        return str(self.step_solver)

    @solving_method.setter
    def solving_method(self, value: str) -> None:
        self.__set_step_solver(value)
        self.__set_second_step_solver(value)

    def refresh_solvers(self):
        self.__set_step_solver(self.solving_method)
        self.__set_second_step_solver(self.solving_method)

    def __set_step_solver(self, value):
        solver_class: Type[Solver] = SolversRegistry.get_by_name(
            solver_name=value, problem=self.problem
        )
        statement = self.problem.statement(self.body)
        if statement is None:
            self.step_solver = None
            return

        self.step_solver = solver_class(
            statement,
            self.body,
            self.time_step,
            getattr(self.problem, "contact_law", None),
            driving_vector=self.driving_vector,
        )

    def __set_second_step_solver(self, value):
        second_solver_class: Type[Solver] = SolversRegistry.get_by_name(
            solver_name=value, problem=self.problem
        )
        statement = self.problem.second_statement(self.body)
        if statement is None:
            self.second_step_solver = None
            return

        self.second_step_solver = second_solver_class(
            statement,
            self.body,
            self.time_step,
            getattr(self.problem, "contact_law_2", None),
            driving_vector=self.driving_vector,
        )

    def solve(self, **kwargs):
        raise NotImplementedError()

    def run(
        self,
        state: State,
        n_steps: int,
        verbose: bool = False,
        products: Optional[List[Product]] = None,
        **kwargs,
    ):
        """
        :param state:
        :param n_steps: number of steps
        :param verbose: show prints
        """
        if products is not None:
            for prod in products:
                state.inject_product(prod)

        for _ in range(n_steps):
            self.step_solver.current_time += self.step_solver.time_step

            solution = self.find_solution(
                state,
                verbose=verbose,
                **kwargs,
            )

            if self.coordinates == "displacement":
                state.set_displacement(
                    solution, update_absement=True, time=self.step_solver.current_time
                )
                self.step_solver.b_vector[:] = state.absement.T.ravel().copy()
                self.step_solver.u_vector[:] = state.displacement.T.ravel().copy()
            elif self.coordinates == "velocity":
                state.set_velocity(
                    solution,
                    update_displacement=True,
                    time=self.step_solver.current_time,
                )
                self.step_solver.u_vector[:] = state.displacement.T.ravel().copy()
                self.step_solver.v_vector[:] = state.velocity.T.ravel().copy()
            else:
                raise ValueError(f"Unknown coordinates: {self.coordinates}")

            self.step_solver.iterate()
            self.done += 1
            print(f"{self.done / self.to_do * 100:.2f}%", end="\r")

    def find_solution(self, state, **kwargs) -> np.ndarray:
        initial_guess = state[self.coordinates].T.ravel().reshape(state.body.mesh.dimension, -1)
        solution = self.step_solver.solve(initial_guess, **kwargs)
        return solution

    def find_solution_uzawa(self, solution, solution_t) -> Tuple[np.ndarray, np.ndarray]:
        # TODO #95
        norm = np.inf
        old_solution = solution.copy().reshape(-1, 1).squeeze()
        old_solution_t = solution_t.copy()
        old_u_vector = self.step_solver.u_vector.copy()
        old_v_vector = self.step_solver.v_vector.copy()
        old_t_vector = self.step_solver.t_vector.copy()
        old_p_vector = self.step_solver.p_vector.copy()
        fuse = 10
        minimum_iter = 5
        while minimum_iter > 0 or norm > 1e-3 and bool(fuse):
            fuse -= 1
            minimum_iter -= 1
            ### iterate
            self.step_solver.statement.update(
                Variables(
                    displacement=old_u_vector,
                    velocity=old_v_vector,
                    temperature=solution_t,
                    electric_potential=solution_t,
                    time_step=self.step_solver.time_step,
                    time=self.step_solver.current_time,
                )
            )
            if isinstance(self.step_solver, SchurComplementOptimization):
                (
                    self.step_solver.node_forces_,
                    self.step_solver.forces_free,
                ) = self.step_solver.recalculate_forces()
            ### end iterate
            solution = self.step_solver.solve(solution)
            ### iterate 2
            u_vector = old_u_vector + self.step_solver.time_step * solution
            self.second_step_solver.statement.update(
                Variables(
                    displacement=u_vector,
                    velocity=solution,
                    temperature=old_t_vector,
                    electric_potential=old_p_vector,
                    time_step=self.second_step_solver.time_step,
                )
            )
            if isinstance(self.second_step_solver, SchurComplementOptimization):
                (
                    self.second_step_solver.node_forces_,
                    self.second_step_solver.forces_free,
                ) = self.second_step_solver.recalculate_forces()
            ### end iterate 2
            solution_t = self.second_step_solver.solve(solution_t)
            norm = (
                np.linalg.norm(solution - old_solution) ** 2
                + np.linalg.norm(old_solution_t - solution_t) ** 2
            ) ** 0.5
            old_solution = solution.copy()
            old_solution_t = solution_t.copy()

        velocity = solution
        self.step_solver.v_vector = velocity.reshape(-1)
        self.step_solver.u_vector = (
            old_u_vector + self.step_solver.time_step * self.step_solver.v_vector
        )
        self.second_step_solver.v_vector = velocity.reshape(-1)
        self.second_step_solver.u_vector = (
            old_u_vector + self.second_step_solver.time_step * self.second_step_solver.v_vector
        )
        self.step_solver.p_vector = solution_t
        self.second_step_solver.p_vector = solution_t
        self.step_solver.t_vector = solution_t
        self.second_step_solver.t_vector = solution_t

        return solution, solution_t


class PoissonSolver(ProblemSolver):
    def __init__(self, problem: Problem, solving_method: str):
        """Solves Poisson problem and saves solution as temperature state.

        :param problem:
        :param solving_method: 'schur', 'optimization', 'direct'
        """
        body_prop = MembraneProperties(
            mass_density=1.0,
            propagation=1.0,
        )
        super().__init__(problem, body_prop)

        _ = TemperatureState(self.body)  # TODO

        self.coordinates = "temperature"
        self.solving_method = solving_method

    # pylint: disable=arguments-differ
    def solve(self, **kwargs) -> TemperatureState:
        state = TemperatureState(self.body)

        self.second_step_solver.t_vector[:] = state.temperature.ravel().copy()
        self.second_step_solver.iterate()

        initial_guess = state[self.coordinates]
        solution = self.second_step_solver.solve(initial_guess=initial_guess, **kwargs)
        state.set_temperature(solution)

        return state


class StaticSolver(ProblemSolver):
    def __init__(self, problem: StaticDisplacementProblem, solving_method: str):
        """Solves general Contact Mechanics problem.

        :param problem:
        :param solving_method: 'schur', 'optimization', 'direct'
        """
        body_prop = ElasticProperties(
            mass_density=1.0,
            mu=problem.mu_coef,
            lambda_=problem.la_coef,
        )
        super().__init__(problem, body_prop)

        _ = State(self.body)  # TODO

        self.coordinates = "displacement"
        self.solving_method = solving_method

    # super class method takes **kwargs, so signatures are consistent
    # pylint: disable=arguments-differ
    def solve(
        self, *, initial_displacement: Callable | Iterable, verbose: bool = False, **kwargs
    ) -> State:
        """
        :param initial_displacement: for the solver
        :param verbose: show prints
        :return: state
        """
        state = State(self.body)
        if hasattr(initial_displacement, "__iter__"):
            state.displacement[:] = initial_displacement
        else:
            state.displacement = initial_displacement(
                self.body.mesh.nodes[: self.body.mesh.nodes_count]
            )

        self.step_solver.u_vector[:] = state.displacement.T.ravel().copy()
        self.step_solver.iterate()
        self.run(state, n_steps=1, verbose=verbose, **kwargs)

        return state


class NonHomogenousSolver(StaticSolver):
    def update_density(self, density: np.ndarray):
        self.body.dynamics.reinitialize_matrices(density)
        self.refresh_solvers()


class QuasistaticRelaxation(ProblemSolver):
    def __init__(self, setup: RelaxationQuasistaticProblem, solving_method: str):
        """Solves general Contact Mechanics problem.

        :param setup:
        :param solving_method: 'schur', 'optimization', 'direct'
        """
        body_prop = ElasticRelaxationProperties(
            mass_density=1.0,
            mu=setup.mu_coef,
            lambda_=setup.la_coef,
            relaxation=setup.relaxation,
        )
        super().__init__(setup, body_prop)

        _ = State(self.body)  # TODO

        self.coordinates = "displacement"
        self.solving_method = solving_method

    # super class method takes **kwargs, so signatures are consistent
    # pylint: disable=arguments-differ
    def solve(
        self,
        *,
        n_steps: int,
        initial_absement: Callable,
        initial_displacement: Callable,
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
        :param initial_absement: for the solver
        :param initial_displacement: for the solver
        :param verbose: show prints
        :return: state
        """
        output_step = (0, *output_step) if output_step else (0, n_steps)  # 0 for diff

        state = State(self.body)
        state.absement[:] = initial_absement(self.body.mesh.nodes[: self.body.mesh.nodes_count])
        state.displacement[:] = initial_displacement(
            self.body.mesh.nodes[: self.body.mesh.nodes_count]
        )

        self.step_solver.b_vector[:] = state.absement.T.ravel().copy()
        self.step_solver.u_vector[:] = state.displacement.T.ravel().copy()
        self.step_solver.iterate()

        output_step = np.diff(output_step)
        results = []
        self.done = 0
        self.to_do = n_steps
        for n in output_step:
            self.run(state, n_steps=n, verbose=verbose, **kwargs)
            results.append(state.copy())

        return results


class TimeDependentSolver(ProblemSolver):
    def __init__(self, problem: TimeDependentDisplacementProblem, solving_method: str):
        """Solves general Contact Mechanics problem.

        :param problem:
        :param solving_method: 'schur', 'optimization', 'direct'
        """
        body_prop = ViscoelasticProperties(
            mass_density=1.0,
            mu=problem.mu_coef,
            lambda_=problem.la_coef,
            theta=problem.th_coef,
            zeta=problem.ze_coef,
        )
        super().__init__(problem, body_prop)

        _ = State(self.body)  # TODO

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
            self.body.mesh.nodes[: self.body.mesh.nodes_count]
        )
        state.velocity[:] = initial_velocity(self.body.mesh.nodes[: self.body.mesh.nodes_count])

        self.step_solver.u_vector[:] = state.displacement.T.ravel().copy()
        self.step_solver.v_vector[:] = state.velocity.T.ravel().copy()
        self.step_solver.iterate()

        output_step = np.diff(output_step)
        results = []
        self.done = 0
        self.to_do = n_steps
        for n in output_step:
            self.run(state, n_steps=n, verbose=verbose, **kwargs)
            results.append(state.copy())

        return results


class TemperatureTimeDependentSolver(ProblemSolver):
    def __init__(self, problem: TemperatureTimeDependentProblem, solving_method: str):
        """Solves general Contact Mechanics problem.

        :param problem:
        :param solving_method: 'schur', 'optimization', 'direct'
        """

        body_prop = ViscoelasticTemperatureProperties(
            mass_density=1.0,
            mu=problem.mu_coef,
            lambda_=problem.la_coef,
            theta=problem.th_coef,
            zeta=problem.ze_coef,
            thermal_expansion=problem.thermal_expansion,
            thermal_conductivity=problem.thermal_conductivity,
        )
        super().__init__(problem, body_prop)

        _ = TemperatureState(self.body)  # TODO

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
        :return: state
        """
        output_step = (0, *output_step) if output_step else (0, n_steps)  # 0 for diff

        state = TemperatureState(self.body)
        state.displacement[:] = initial_displacement(
            self.body.mesh.nodes[: self.body.mesh.nodes_count]
        )
        state.velocity[:] = initial_velocity(self.body.mesh.nodes[: self.body.mesh.nodes_count])
        state.temperature[:] = initial_temperature(
            self.body.mesh.nodes[: self.body.mesh.nodes_count]
        )

        solution = state.velocity.reshape(2, -1)
        solution_t = state.temperature

        self.step_solver.u_vector[:] = state.displacement.T.ravel().copy()
        self.step_solver.v_vector[:] = state.velocity.T.ravel().copy()
        self.step_solver.t_vector[:] = state.temperature.T.ravel().copy()

        self.second_step_solver.u_vector[:] = state.displacement.T.ravel().copy()
        self.second_step_solver.v_vector[:] = state.velocity.T.ravel().copy()
        self.second_step_solver.t_vector[:] = state.temperature.T.ravel().copy()

        self.step_solver.iterate()
        self.second_step_solver.iterate()

        output_step = np.diff(output_step)
        done = 0
        for n in output_step:
            for _ in range(n):
                done += 1
                print(f"{done/n_steps*100:.2f}%", end="\r")
                self.step_solver.current_time += self.step_solver.time_step
                self.second_step_solver.current_time += self.second_step_solver.time_step

                solution, solution_t = self.find_solution_uzawa(solution, solution_t)

                if self.coordinates == "velocity":
                    state.set_velocity(
                        solution[:],
                        update_displacement=True,
                        time=self.step_solver.current_time,
                    )
                    state.set_temperature(solution_t)
                else:
                    raise ValueError(f"Unknown coordinates: {self.coordinates}")
            yield state.copy()


class PiezoelectricTimeDependentSolver(ProblemSolver):
    def __init__(self, problem: PiezoelectricTimeDependentProblem, solving_method: str):
        """Solves general Contact Mechanics problem.

        :param solving_method: 'schur', 'optimization', 'direct'
        """
        body_prop = ViscoelasticPiezoelectricProperties(
            mass_density=0.1,
            mu=problem.mu_coef,
            lambda_=problem.la_coef,
            theta=problem.th_coef,
            zeta=problem.ze_coef,
            piezoelectricity=problem.piezoelectricity,
            permittivity=problem.permittivity,
        )
        super().__init__(problem, body_prop)

        _ = PiezoelectricState(self.body)  # TODO

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
        :return: state
        """
        output_step = (0, *output_step) if output_step else (0, n_steps)  # 0 for diff

        state = PiezoelectricState(self.body)
        state.displacement[:] = initial_displacement(
            self.body.mesh.nodes[: self.body.mesh.nodes_count]
        )
        state.velocity[:] = initial_velocity(self.body.mesh.nodes[: self.body.mesh.nodes_count])
        state.electric_potential[:] = initial_electric_potential(
            self.body.mesh.nodes[: self.body.mesh.nodes_count]
        )

        solution = state.velocity.reshape(2, -1)
        solution_t = state.electric_potential

        self.step_solver.u_vector[:] = state.displacement.T.ravel().copy()
        self.step_solver.v_vector[:] = state.velocity.T.ravel().copy()
        self.step_solver.p_vector[:] = state.electric_potential.T.ravel().copy()

        self.second_step_solver.u_vector[:] = state.displacement.T.ravel().copy()
        self.second_step_solver.v_vector[:] = state.velocity.T.ravel().copy()
        self.second_step_solver.p_vector[:] = state.electric_potential.T.ravel().copy()

        self.step_solver.iterate()
        self.second_step_solver.iterate()

        output_step = np.diff(output_step)
        results = []
        done = 0
        for n in output_step:
            for _ in range(n):
                done += 1
                print(f"{done/n_steps*100:.2f}%", end="\r")
                self.step_solver.current_time += self.step_solver.time_step
                self.second_step_solver.current_time += self.second_step_solver.time_step

                solution, solution_t = self.find_solution_uzawa(solution, solution_t)

                if self.coordinates == "velocity":
                    state.set_velocity(
                        solution[:],
                        update_displacement=True,
                        time=self.step_solver.current_time,
                    )
                    state.set_electric_potential(solution_t)
                else:
                    raise ValueError(f"Unknown coordinates: {self.coordinates}")
            results.append(state.copy())

        return results


class WaveSolver(ProblemSolver):
    def __init__(
        self,
        problem: WaveProblem,
        solving_method: str,
    ):
        """Solves general Contact Mechanics problem.

        :param solving_method: 'schur', 'optimization', 'direct'
        """
        body_prop = MembraneProperties(mass_density=1, propagation=problem.propagation)
        super().__init__(problem, body_prop)

        _ = State(self.body)  # TODO

        self.coordinates = "velocity"
        self.driving_vector = True
        self.solving_method = solving_method

    # super class method takes **kwargs, so signatures are consistent
    # pylint: disable=arguments-differ
    def solve(
        self,
        *,
        n_steps: int,
        initial_displacement: Callable,
        initial_velocity: Callable,
        state: Optional[State] = None,
        output_step: Optional[iter] = None,
        **kwargs,
    ) -> List[State]:
        """
        :param n_steps: number of time-step in simulation
        :param output_step: from which time-step we want to get copy of State,
                            default (n_steps-1,)
                            example: for Setup.time-step = 2, n_steps = 10,
                            output_step = (2, 6, 9) we get 3 shared copy of
                            State for time-steps 4, 12 and 18
        :param initial_displacement: for the solver
        :param initial_velocity: for the solver
        :param state: if present, simulation starts from this state,
                      use this to resume stopped simulation.
        :return: state
        """
        output_step = (0, *output_step) if output_step else (0, n_steps)  # 0 for diff

        if state is None:
            state = State(self.body)
            state.displacement[:] = initial_displacement(
                self.body.mesh.nodes[: self.body.mesh.nodes_count]
            )
            state.velocity[:] = initial_velocity(self.body.mesh.nodes[: self.body.mesh.nodes_count])

        self.step_solver.current_time = state.time

        self.step_solver.u_vector[:] = state.displacement.T.ravel().copy()
        self.step_solver.v_vector[:] = state.velocity.T.ravel().copy()
        self.step_solver.iterate()

        output_step = np.diff(output_step)
        results = []
        self.done = 0
        self.to_do = n_steps
        for n in output_step:
            self.run(state, n_steps=n, **kwargs)
            results.append(state.copy())

        return results
