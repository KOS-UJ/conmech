from ctypes import ArgumentError
from typing import Optional

import jax
import jax.experimental
import jax.numpy as jnp
import jax.scipy
import jax.scipy.optimize

# import jaxopt
import numpy as np

from conmech.helpers import cmh, jxh, nph
from conmech.helpers.tmh import Timer
from conmech.scene.body_forces import energy
from conmech.scene.energy_functions import EnergyFunctions
from conmech.scene.scene import Scene
from conmech.scene.scene_temperature import SceneTemperature
from conmech.solvers.algorithms.lbfgs import minimize_lbfgs

# from jax._src.scipy.optimize.bfgs import minimize_bfgs
# import tensorflow_probability as tfp


class Calculator:
    @staticmethod
    def minimize_jax(
        function, initial_vector: np.ndarray, args, hes_inv, verbose: bool = True
    ) -> np.ndarray:
        assert cmh.get_from_os("ENV_READY")

        # state1 = minimize_lbfgs(
        #     fun=function, hes_inv=hes_inv, args=args, x0=jnp.asarray(initial_vector)
        # )

        range_factor = args.time_step**2

        disp_by_factor_0 = (
            nph.acceleration_to_displacement(jnp.asarray(initial_vector), args) / range_factor
        )

        state = cmh.profile(
            lambda: minimize_lbfgs(
                fun=function,
                hes_inv=hes_inv,
                args=args,
                x0=disp_by_factor_0,
                xtol=1e-03,
            ),
            baypass=True,
        )

        if False:  # cmh.get_from_os("JAX_ENABLE_X64"):
            assert state.converged
        else:
            if verbose and not state.converged:
                if state.status == 5:
                    cmh.Console.print_warning("Linesearch error")
                elif state.status == 1:
                    cmh.Console.print_fail("Maxiter error")
                else:
                    cmh.Console.print_fail(f"Status: {state.status}")
        # Validate https://github.com/google/jax/issues/6898
        # return np.asarray(state.x_k)  # , state
        disp_by_factor = state.x_k
        return nph.displacement_to_acceleration(np.asarray(disp_by_factor * range_factor), args)

    @staticmethod
    def solve_temperature_normalized_function(
        scene: SceneTemperature, normalized_acceleration: np.ndarray, initial_t
    ):
        assert cmh.get_from_os("ENV_READY")
        normalized_rhs = scene.get_normalized_t_rhs_jax(normalized_acceleration)
        matrix = scene.solver_cache.lhs_temperature_sparse_jax
        vector = normalized_rhs
        initial_point = initial_t

        solver = jax.jit(jax.scipy.sparse.linalg.cg)
        t_vector, _ = solver(A=matrix, b=vector, x0=initial_point)
        return np.array(t_vector)

    @staticmethod
    def solve_acceleration_normalized_function(scene, temperature=None, initial_a=None):
        assert cmh.get_from_os("ENV_READY")
        # normalized_a_vector, _ = scipy.sparse.linalg.cg(A=A, b=b)

        matrix = scene.solver_cache.lhs_sparse_jax
        vector = scene.get_normalized_rhs_jax(temperature)
        initial_point = jnp.array(nph.stack_column(initial_a)) if initial_a is not None else None

        if scene.simulation_config.use_lhs_preconditioner:
            preconditioner = scene.solver_cache.lhs_preconditioner_jax
        else:
            preconditioner = None

        # A is symetric and positive definite
        # A_ = A.get().todense()
        # np.allclose(A_, A_.T)
        # np.all(np.linalg.eigvals(A_) > 0)
        # M_ = M.get().todense()
        # np.linalg.cond(A_) ~ 646
        # np.linalg.cond(M_ @ A_) ~ 421

        solver = jax.jit(jax.scipy.sparse.linalg.cg)
        normalized_a_vector, _ = cmh.profile(
            lambda: solver(A=matrix, b=vector, x0=initial_point, M=preconditioner),
            baypass=True,
        )
        normalized_a = np.array(nph.unstack(normalized_a_vector, scene.dimension))
        # assert info == 0
        # assert np.allclose(A @ normalized_a_vector_jax - b.reshape(-1), 0)

        normalized_cleaned_a = scene.clean_acceleration(normalized_a)
        cleaned_a = Calculator.denormalize(scene, normalized_cleaned_a)
        return cleaned_a

    @staticmethod
    def solve(
        scene: Scene,
        energy_functions: EnergyFunctions,
        initial_a: Optional[np.ndarray] = None,
        initial_t: Optional[np.ndarray] = None,
        timer: Timer = Timer(),
    ) -> np.ndarray:
        with timer["calc_optimize"]:
            normalized_a = Calculator.solve_acceleration_normalized(
                scene, energy_functions, initial_a=initial_a, timer=timer
            )
        normalized_cleaned_a = scene.clean_acceleration(normalized_a)
        with timer["calc_denormalize"]:
            cleaned_a = Calculator.denormalize(scene, normalized_cleaned_a)
        return cleaned_a, initial_t

    @staticmethod
    def solve_with_temperature(
        scene: SceneTemperature,
        energy_functions: EnergyFunctions,
        initial_a: Optional[np.ndarray] = None,
        initial_t: Optional[np.ndarray] = None,
        timer: Timer = Timer(),
    ):
        _ = timer
        uzawa = False
        max_iter = 10
        i = 0
        normalized_a = None
        temperature = scene.t_old
        last_normalized_a, normalized_a, last_t = np.empty(0), np.empty(0), np.empty(0)
        while (
            i < 2
            or not np.allclose(last_normalized_a, normalized_a)
            and not np.allclose(last_t, temperature)
        ):
            last_normalized_a, last_t = normalized_a, temperature
            normalized_a = Calculator.solve_acceleration_normalized(
                scene, energy_functions, temperature, initial_a
            )
            temperature = Calculator.solve_temperature_normalized(scene, normalized_a, initial_t)
            i += 1
            if i >= max_iter:
                raise ArgumentError(f"Uzawa algorithm: maximum of {max_iter} iterations exceeded")
            if uzawa is False:
                break

        normalized_cleaned_a = scene.clean_acceleration(normalized_a)
        cleaned_a = Calculator.denormalize(scene, normalized_cleaned_a)
        cleaned_t = Calculator.clean_temperature(scene, temperature)
        return cleaned_a, cleaned_t

    @staticmethod
    def solve_temperature(
        setting: SceneTemperature, normalized_acceleration: np.ndarray, initial_t
    ):
        t = Calculator.solve_temperature_normalized(setting, normalized_acceleration, initial_t)
        cleaned_t = Calculator.clean_temperature(setting, t)
        return cleaned_t

    @staticmethod
    def solve_acceleration_normalized(
        scene: Scene,
        energy_functions: EnergyFunctions,
        temperature=None,
        initial_a: Optional[np.ndarray] = None,
        timer: Timer = Timer(),
    ) -> np.ndarray:
        if scene.simulation_config.use_linear_solver:
            return Calculator.solve_acceleration_normalized_function(
                scene=scene, temperature=temperature, initial_a=initial_a
            )
        # TODO: #62 repeat with optimization if collision in this round
        # if not scene.is_colliding():
        return Calculator.solve_acceleration_normalized_optimization_jax(
            scene=scene,
            energy_functions=energy_functions,
            temperature=temperature,
            initial_a=initial_a,
            timer=timer,
        )

    @staticmethod
    def solve_temperature_normalized(
        scene: SceneTemperature,
        normalized_acceleration: np.ndarray,
        initial_t: Optional[np.ndarray] = None,
    ) -> np.ndarray:
        # TODO: #62 repeat with optimization if collision in this round
        # if not scene.is_colliding():
        if scene.simulation_config.use_linear_solver:
            return Calculator.solve_temperature_normalized_function(
                scene, normalized_acceleration, initial_t
            )
        return Calculator.solve_temperature_normalized_optimization(
            scene, normalized_acceleration, initial_t
        )

    @staticmethod
    def solve_acceleration_normalized_optimization_jax(
        scene: Scene,
        energy_functions: EnergyFunctions,
        temperature=None,
        initial_a=None,
        timer: Timer = Timer(),
    ):
        if initial_a is None:
            initial_a_vector = np.zeros(scene.nodes_count * scene.dimension)
        else:
            initial_a_vector = nph.stack(initial_a)

        with timer["__get_energy_obstacle_args"]:
            args = cmh.profile(
                lambda: scene.get_energy_obstacle_args_for_jax(energy_functions, temperature),
                baypass=True,
            )
        hes_inv = (
            None
            if not scene.simulation_config.use_lhs_preconditioner
            else scene.solver_cache.lhs_preconditioner_jax
        )
        with timer["__minimize_jax"]:
            normalized_a_vector_np = cmh.profile(
                lambda: Calculator.minimize_jax(
                    function=energy_functions.get_energy_function(scene),
                    # solver= energy_functions.get_solver(scene),
                    initial_vector=initial_a_vector,
                    args=args,
                    hes_inv=hes_inv,
                ),
                baypass=True,
            )

        normalized_a_vector = normalized_a_vector_np.reshape(-1, 1)
        return nph.unstack(normalized_a_vector, scene.dimension)

    @staticmethod
    def solve_temperature_normalized_optimization(
        scene: SceneTemperature,
        normalized_a: np.ndarray,
        initial_t: Optional[np.ndarray] = None,
    ):
        if initial_t is None:
            initial_t_vector = np.zeros(scene.nodes_count)
        else:
            initial_t_vector = nph.stack(initial_t)

        normalized_t_rhs = scene.get_normalized_t_rhs_jax(normalized_a)

        cost_function = jax.jit(
            lambda x, args: energy(
                nph.unstack(x, 1),
                scene.solver_cache.lhs_temperature_sparse_jax,
                normalized_t_rhs,
            )
        )

        normalized_t_vector = Calculator.minimize_jax(
            function=cost_function,
            initial_vector=initial_t_vector,
            args=None,
            hes_inv=None,
        )
        result = nph.unstack(normalized_t_vector, 1)
        return result

    @staticmethod
    def solve_temperature_normalized_optimization_schur(
        scene: SceneTemperature,
        normalized_a: np.ndarray,
        initial_t: Optional[np.ndarray] = None,
    ):
        _ = initial_t
        initial_t_boundary_vector = np.zeros(scene.boundary_nodes_count)

        (
            cost_function,
            normalized_t_rhs_free,
        ) = scene.get_normalized_energy_temperature_np(normalized_a)
        boundary_t_vector_np = Calculator.minimize_jax(
            cost_function,
            initial_t_boundary_vector,
            args=None,
            hes_inv=None,
        )

        boundary_t_vector = boundary_t_vector_np.reshape(-1, 1)
        t_vector = Calculator.complete_t_vector(scene, normalized_t_rhs_free, boundary_t_vector)
        return t_vector

    @staticmethod
    def clean_temperature(scene, temperature):
        _ = scene
        return temperature if temperature is not None else None

    @staticmethod
    def denormalize(setting, normalized_cleaned_a):
        return setting.denormalize_rotate(normalized_cleaned_a)

    @staticmethod
    def complete_a_vector(scene, normalized_rhs_free, a_contact_vector):
        # a_independent_vector = scene.solver_cache.free_x_free_inverted @ (
        #     normalized_rhs_free - (scene.solver_cache.free_x_contact @ a_contact_vector)
        # )
        s1 = normalized_rhs_free - (scene.solver_cache.free_x_contact @ a_contact_vector)
        a_independent_vector = jxh.solve_linear_jax(
            matrix=scene.solver_cache.free_x_free, vector=s1
        )

        normalized_a = np.vstack(
            (
                nph.unstack(a_contact_vector, scene.dimension),
                nph.unstack(a_independent_vector, scene.dimension),
            )
        )
        return nph.stack(normalized_a)

    @staticmethod
    def complete_t_vector(scene: SceneTemperature, normalized_t_rhs_free, t_contact_vector):
        t_independent_vector = scene.solver_cache.temperature_free_x_free_inv @ (
            normalized_t_rhs_free
            - (scene.solver_cache.temperature_free_x_contact @ t_contact_vector)
        )

        return np.vstack((t_contact_vector, t_independent_vector))
