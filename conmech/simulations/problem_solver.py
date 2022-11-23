"""
General solver for Contact Mechanics problem.
"""
from typing import Callable, List, Optional, Tuple

import numpy as np

from conmech.dynamics.dynamics import DynamicsConfiguration
from conmech.dynamics.statement import (
    StaticDisplacementStatement,
    QuasistaticVelocityStatement,
    DynamicVelocityWithTemperatureStatement,
    TemperatureStatement,
    PiezoelectricStatement,
    DynamicVelocityStatement,
    QuasistaticVelocityWithPiezoelectricStatement, Variables,
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
    Dynamic as DynamicProblem,
    TimeDependent as TimeDependentProblem,
)
from conmech.scenarios.problems import Problem
from conmech.scenarios.problems import Quasistatic as QuasistaticProblem
from conmech.scenarios.problems import Static as StaticProblem
from conmech.scenarios.problems import TemperatureDynamic as TemperatureDynamicProblem
from conmech.scenarios.problems import TemperatureTimeDependent as TemperatureTimeDependentProblem
from conmech.scenarios.problems import PiezoelectricQuasistatic as PiezoelectricQuasistaticProblem
from conmech.scenarios.problems import (
    PiezoelectricTimeDependent as PiezoelectricTimeDependentProblem,
)
from conmech.scene.body_forces import BodyForces
from conmech.solvers import Solvers, SchurComplement
from conmech.solvers.solver import Solver
from conmech.solvers.validator import Validator
from conmech.state.state import State, TemperatureState, PiezoelectricState


class ProblemSolver:
    def __init__(self, setup: Problem, body_properties: BodyProperties):
        """Solves general Contact Mechanics problem.

        :param setup:
        :param body_properties:
        """
        if isinstance(setup, TimeDependentProblem):
            time_step = setup.time_step
        else:
            time_step = 0

        grid_width = (setup.grid_height / setup.elements_number[0]) * setup.elements_number[1]

        self.body = BodyForces(
            mesh_prop=MeshProperties(
                dimension=2,
                mesh_type=setup.mesh_type,
                mesh_density=[setup.elements_number[1], setup.elements_number[0]],
                scale=[float(grid_width), float(setup.grid_height)],
            ),
            body_prop=body_properties,
            schedule=Schedule(time_step=time_step, final_time=0.0),
            boundaries_description=setup.boundaries,
            dynamics_config=DynamicsConfiguration(
                normalize_by_rotation=False,
                create_in_subprocess=False,
                with_lhs=False,
                with_schur=False,
            ),
        )
        self.body.set_permanent_forces_by_functions(
            inner_forces_function=setup.inner_forces, outer_forces_function=setup.outer_forces
        )
        self.setup = setup

        self.coordinates = None
        self.step_solver: Optional[Solver] = None
        self.second_step_solver: Optional[Solver] = None
        self.validator: Optional[Validator] = None

    @property
    def solving_method(self):
        return str(self.step_solver)

    @solving_method.setter
    def solving_method(self, value):
        solver_class = Solvers.get_by_name(solver_name=value, problem=self.setup)
        second_solver_class = Solvers.get_by_name(solver_name=value, problem=self.setup)

        # TODO: #65 fixed solvers to avoid: th_coef, ze_coef = mu_coef, la_coef
        if isinstance(self.setup, StaticProblem):
            statement = StaticDisplacementStatement(self.body)
            time_step = 0
        elif isinstance(self.setup, (QuasistaticProblem, DynamicProblem)):
            if isinstance(self.setup, PiezoelectricQuasistaticProblem):
                statement = QuasistaticVelocityWithPiezoelectricStatement(self.body)
            elif isinstance(self.setup, QuasistaticProblem):
                statement = QuasistaticVelocityStatement(self.body)
            elif isinstance(self.setup, TemperatureDynamicProblem):
                statement = DynamicVelocityWithTemperatureStatement(self.body)
            else:
                statement = DynamicVelocityStatement(self.body)
            time_step = self.setup.time_step
        else:
            raise ValueError(f"Unknown problem class: {self.setup.__class__}")

        self.step_solver = solver_class(
            statement,
            self.body,
            time_step,
            self.setup.contact_law,
            self.setup.friction_bound,
        )
        if isinstance(self.setup, TemperatureTimeDependentProblem):
            self.second_step_solver = solver_class(
                TemperatureStatement(self.body),
                self.body,
                time_step,
                self.setup.contact_law,
                self.setup.friction_bound,
            )
        elif isinstance(self.setup, PiezoelectricTimeDependentProblem):
            self.second_step_solver = second_solver_class(
                PiezoelectricStatement(self.body),
                self.body,
                time_step,
                self.setup.contact_law,
                self.setup.friction_bound,
            )
        else:
            self.second_step_solver = None
        self.validator = Validator(self.step_solver)

    def solve(self, **kwargs):
        raise NotImplementedError()

    def run(self, state, n_steps: int, verbose: bool = False, **kwargs):
        """
        :param state:
        :param n_steps: number of steps
        :param verbose: show prints
        :return: state
        """
        for _ in range(n_steps):
            self.step_solver.current_time += self.step_solver.time_step
            solution = state.velocity.reshape(2, -1)
            solution = self.find_solution(
                state,
                solution,
                self.validator,
                verbose=verbose,
                **kwargs,
            )

            if self.coordinates == "displacement":
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

    def find_solution(self, state, solution, validator, *, verbose=False, **kwargs) -> np.ndarray:
        quality = 0
        # solution = state[self.coordinates].reshape(2, -1)  # TODO #23
        solution = self.step_solver.solve(solution, **kwargs)
        self.step_solver.iterate(solution)
        quality = validator.check_quality(state, solution, quality)
        self.print_iteration_info(quality, validator.error_tolerance, verbose)
        return solution

    def find_solution_uzawa(
        self, state, solution, solution_t, *, verbose=False
    ) -> Tuple[np.ndarray, np.ndarray]:
        norm = np.inf
        old_solution = solution.copy().reshape(-1, 1).squeeze()
        old_solution_t = solution_t.copy()
        old_u_vector = self.step_solver.u_vector.copy()
        old_v_vector = self.step_solver.v_vector.copy()
        old_t_vector = self.step_solver.t_vector.copy()
        old_p_vector = self.step_solver.p_vector.copy()
        fuse = 10
        minimum_iter = 2
        while minimum_iter > 0 or norm > 1e-3 and bool(fuse):
            fuse -= 1
            minimum_iter -= 1
            ### iterate
            self.step_solver.statement.update(
                Variables(
                    displacement=old_u_vector,
                    velocity=old_v_vector,
                    temperature=solution_t,
                    electric_potential=solution_t,  # TODO
                    time_step=self.step_solver.time_step,
                )
            )
            if isinstance(self.step_solver, SchurComplement):
                self.step_solver.node_forces_, self.step_solver.forces_free = self.step_solver.recalculate_forces()
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
            if isinstance(self.second_step_solver, SchurComplement):
                self.second_step_solver.node_forces_, self.second_step_solver.forces_free = self.second_step_solver.recalculate_forces()
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
        self.step_solver.u_vector = old_u_vector + self.step_solver.time_step * self.step_solver.v_vector
        self.second_step_solver.v_vector = velocity.reshape(-1)
        self.second_step_solver.u_vector = old_u_vector + self.second_step_solver.time_step * self.second_step_solver.v_vector
        self.step_solver.p_vector = solution_t  # FIXME
        self.second_step_solver.p_vector = solution_t  # FIXME
        self.step_solver.t_vector = solution_t  # FIXME
        self.second_step_solver.t_vector = solution_t  # FIXME

        return solution, solution_t

    @staticmethod
    def print_iteration_info(quality: float, error_tolerance: float, verbose: bool):
        qualitative = quality > error_tolerance
        sign = ">" if qualitative else "<"
        end = "." if qualitative else ", trying again..."
        if verbose:
            print(f"quality = {quality} {sign} {error_tolerance}{end}")


class Static(ProblemSolver):
    def __init__(self, setup: StaticProblem, solving_method: str):
        """Solves general Contact Mechanics problem.

        :param setup:
        :param solving_method: 'schur', 'optimization', 'direct'
        """
        body_prop = StaticBodyProperties(
            mass_density=1.0,
            mu=setup.mu_coef,
            lambda_=setup.la_coef,
        )
        super().__init__(setup, body_prop)

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
            self.body.mesh.initial_nodes[: self.body.mesh.nodes_count]
        )

        solution = state.displacement.reshape(2, -1)

        self.step_solver.u_vector[:] = state.displacement.ravel().copy()

        self.run(state, n_steps=1, verbose=verbose, **kwargs)

        return state


class TimeDependent(ProblemSolver):
    def __init__(self, setup: TimeDependentProblem, solving_method: str):
        """Solves general Contact Mechanics problem.

        :param setup:
        :param solving_method: 'schur', 'optimization', 'direct'
        """
        body_prop = TimeDependentBodyProperties(
            mass_density=1.0,
            mu=setup.mu_coef,
            lambda_=setup.la_coef,
            theta=setup.th_coef,
            zeta=setup.ze_coef,
        )
        super().__init__(setup, body_prop)

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

        state = State(self.body)
        state.displacement[:] = initial_displacement(
            self.body.mesh.initial_nodes[: self.body.mesh.nodes_count]
        )
        state.velocity[:] = initial_velocity(
            self.body.mesh.initial_nodes[: self.body.mesh.nodes_count]
        )

        self.step_solver.u_vector[:] = state.displacement.ravel().copy()
        self.step_solver.v_vector[:] = state.velocity.ravel().copy()

        output_step = np.diff(output_step)
        results = []
        done = 0
        for n in output_step:
            done += n
            print(f"{done / n_steps * 100:.2f}%", end="\r")
            self.run(state, n_steps=n, verbose=verbose)
            results.append(state.copy())

        return results


class TemperatureTimeDependent(ProblemSolver):
    def __init__(self, setup: TemperatureTimeDependentProblem, solving_method: str):
        """Solves general Contact Mechanics problem.

        :param setup:
        :param solving_method: 'schur', 'optimization', 'direct'
        """

        body_prop = TimeDependentTemperatureBodyProperties(
            mass_density=1.0,
            mu=setup.mu_coef,
            lambda_=setup.la_coef,
            theta=setup.th_coef,
            zeta=setup.ze_coef,
            thermal_expansion=setup.thermal_expansion,
            thermal_conductivity=setup.thermal_conductivity,
        )
        super().__init__(setup, body_prop)

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
            self.body.mesh.initial_nodes[: self.body.mesh.nodes_count]
        )
        state.velocity[:] = initial_velocity(
            self.body.mesh.initial_nodes[: self.body.mesh.nodes_count]
        )
        state.temperature[:] = initial_temperature(
            self.body.mesh.initial_nodes[: self.body.mesh.nodes_count]
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
        done = 0
        for n in output_step:
            for _ in range(n):
                done += 1
                print(f"{done/n_steps*100:.2f}%", end="\r")
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
            yield state.copy()


class PiezoelectricTimeDependent(ProblemSolver):
    def __init__(self, setup: PiezoelectricTimeDependentProblem, solving_method: str):
        """Solves general Contact Mechanics problem.

        :param setup:
        :param solving_method: 'schur', 'optimization', 'direct'
        """
        body_prop = TimeDependentPiezoelectricBodyProperties(
            mass_density=0.1,
            mu=setup.mu_coef,
            lambda_=setup.la_coef,
            theta=setup.th_coef,
            zeta=setup.ze_coef,
            piezoelectricity=setup.piezoelectricity,
            permittivity=setup.permittivity,
        )
        super().__init__(setup, body_prop)

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
            self.body.mesh.initial_nodes[: self.body.mesh.nodes_count]
        )
        state.velocity[:] = initial_velocity(
            self.body.mesh.initial_nodes[: self.body.mesh.nodes_count]
        )
        state.electric_potential[:] = initial_electric_potential(
            self.body.mesh.initial_nodes[: self.body.mesh.nodes_count]
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
        done = 0
        for n in output_step:
            for _ in range(n):
                done += 1
                print(f"{done/n_steps*100:.2f}%", end="\r")
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
