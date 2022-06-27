from ctypes import ArgumentError
from typing import Callable, Optional

import cupy as cp
import cupy.linalg
import cupyx.scipy.sparse
import cupyx.scipy.sparse.linalg
import jax
import jax.experimental
import jax.numpy as jnp
import jax.scipy
import jax.scipy.optimize
import numpy as np
import scipy.optimize
import scipy.sparse.linalg
from psutil import MACOS

from conmech.helpers import jxh, nph
from conmech.scene.scene import Scene
from conmech.scene.scene_temperature import SceneTemperature
from deep_conmech.scene.scene_randomized import SceneRandomized


class Calculator:
    @staticmethod
    def minimize_np(
        function: Callable[[np.ndarray], np.ndarray], initial_vector: np.ndarray
    ) -> np.ndarray:
        return scipy.optimize.minimize(
            function,
            initial_vector,
            method="L-BFGS-B",
        ).x

    @staticmethod
    def minimize_jax(
        function: Callable[[np.ndarray], np.ndarray], initial_vector: np.ndarray
    ) -> np.ndarray:
        x0 = jnp.asarray(initial_vector)
        # jacobian = jax.jacfwd(jax_function)
        result = jax.scipy.optimize.minimize(
            function, x0, method="l-bfgs-experimental-do-not-rely-on-this"
        )
        # jax.jit()
        # "l-bfgs-experimental-do-not-rely-on-this"
        # BFGS")
        return np.asarray(result.x)

    @staticmethod
    def solve(setting: Scene, initial_a: Optional[np.ndarray] = None) -> np.ndarray:
        cleaned_a, _ = Calculator.solve_all(setting, initial_a)
        return cleaned_a

    @staticmethod
    def solve_all(setting: Scene, initial_a: Optional[np.ndarray] = None):
        normalized_a = Calculator.solve_acceleration_normalized(setting, initial_a=initial_a)
        normalized_cleaned_a = Calculator.clean_acceleration(setting, normalized_a)
        cleaned_a = Calculator.denormalize(setting, normalized_cleaned_a)
        return cleaned_a, normalized_cleaned_a

    @staticmethod
    def solve_with_temperature(
        scene: SceneTemperature,
        initial_a: Optional[np.ndarray] = None,
        initial_t: Optional[np.ndarray] = None,
    ):
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
            normalized_a = Calculator.solve_acceleration_normalized(scene, temperature, initial_a)
            temperature = Calculator.solve_temperature_normalized(scene, normalized_a, initial_t)
            i += 1
            if i >= max_iter:
                raise ArgumentError(f"Uzawa algorithm: maximum of {max_iter} iterations exceeded")
            if uzawa is False:
                break

        normalized_cleaned_a = Calculator.clean_acceleration(scene, normalized_a)
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
        setting: Scene, temperature=None, initial_a: Optional[np.ndarray] = None
    ) -> np.ndarray:
        # TODO: #62 repeat with optimization if collision in this round
        if False:  # setting.is_colliding():
            return Calculator.solve_acceleration_normalized_optimization_jax(
                setting, temperature, initial_a
            )
        return Calculator.solve_acceleration_normalized_function(
            setting=setting, temperature=temperature, initial_a=initial_a
        )

    @staticmethod
    def solve_temperature_normalized(
        setting: SceneTemperature,
        normalized_acceleration: np.ndarray,
        initial_t: Optional[np.ndarray] = None,
    ) -> np.ndarray:
        # TODO: #62 repeat with optimization if collision in this round
        if setting.is_colliding():
            return Calculator.solve_temperature_normalized_optimization(
                setting, normalized_acceleration, initial_t
            )
        return Calculator.solve_temperature_normalized_function(
            setting, normalized_acceleration, initial_t
        )

    @staticmethod
    def solve_temperature_normalized_function(
        setting: SceneTemperature, normalized_acceleration: np.ndarray, initial_t
    ):
        _ = initial_t
        normalized_rhs = setting.get_normalized_t_rhs_np(normalized_acceleration)
        t_vector = np.linalg.solve(setting.solver_cache.lhs_temperature, normalized_rhs)
        return t_vector

    @staticmethod
    def solve_acceleration_normalized_function_np(setting, temperature=None, initial_a=None):
        _ = initial_a
        normalized_rhs = setting.get_normalized_rhs_np(temperature)

        A = setting.solver_cache.lhs_sparse
        b = normalized_rhs

        normalized_a_vector, _ = scipy.sparse.linalg.cg(A=A, b=b)
        return nph.unstack(normalized_a_vector, setting.dimension)

    @staticmethod
    def solve_acceleration_normalized_function(setting, temperature=None, initial_a=None):
        normalized_rhs = setting.get_normalized_rhs_cp(temperature)

        A = setting.solver_cache.lhs_sparse_cp
        b = normalized_rhs
        x0 = cp.array(nph.stack_column(initial_a)) if initial_a is not None else None

        M = setting.solver_cache.lhs_preconditioner_cp

        # A is symetric and positive definite
        # A_ = A.get().todense()
        # np.allclose(A_, A_.T)
        # np.all(np.linalg.eigvals(A_) > 0)
        # M_ = M.get().todense()
        # np.linalg.cond(A_) ~ 646
        # np.linalg.cond(M_ @ A_) ~ 421

        normalized_a_vector_cp, _ = cupyx.scipy.sparse.linalg.cg(A=A, b=b, x0=x0, M=M)
        # assert info == 0
        # assert np.allclose(A @ normalized_a_vector_cp - b.reshape(-1), 0)
        normalized_a_vector = normalized_a_vector_cp.get()
        return nph.unstack(normalized_a_vector, setting.dimension)

    @staticmethod
    def solve_all_acceleration_normalized_function(setting, temperature=None, initial_a=None):
        normalized_a = Calculator.solve_acceleration_normalized_function(
            setting, temperature, initial_a
        )
        normalized_cleaned_a = Calculator.clean_acceleration(setting, normalized_a)
        cleaned_a = Calculator.denormalize(setting, normalized_cleaned_a)
        return cleaned_a, normalized_cleaned_a

    @staticmethod
    def get_acceleration_energy(setting, acceleration):
        initial_a_boundary_vector = nph.stack_column(acceleration[setting.boundary_indices])

        cost_function, _ = setting.get_normalized_energy_obstacle_jax()
        energy = cost_function(initial_a_boundary_vector)
        return energy

    @staticmethod
    def solve_acceleration_normalized_optimization_jax(setting, temperature=None, initial_a=None):
        if initial_a is None:
            initial_a_boundary_vector = np.zeros(setting.boundary_nodes_count * setting.dimension)
        else:
            initial_a_boundary_vector = nph.stack(initial_a[setting.boundary_indices])

        cost_function, normalized_rhs_free = setting.get_normalized_energy_obstacle_jax(temperature)
        normalized_boundary_a_vector_np = Calculator.minimize_jax(
            cost_function, initial_a_boundary_vector
        )

        normalized_boundary_a_vector = normalized_boundary_a_vector_np.reshape(-1, 1)
        normalized_a_vector = Calculator.complete_a_vector(
            setting, normalized_rhs_free, normalized_boundary_a_vector
        )

        return nph.unstack(normalized_a_vector, setting.dimension)

    @staticmethod
    def solve_temperature_normalized_optimization(
        setting: SceneTemperature,
        normalized_a: np.ndarray,
        initial_t_vector: Optional[np.ndarray] = None,
    ):
        _ = initial_t_vector
        initial_t_boundary_vector = np.zeros(setting.boundary_nodes_count)

        (
            cost_function,
            normalized_t_rhs_free,
        ) = setting.get_normalized_energy_temperature_np(normalized_a)
        boundary_t_vector_np = Calculator.minimize_np(cost_function, initial_t_boundary_vector)

        boundary_t_vector = boundary_t_vector_np.reshape(-1, 1)
        t_vector = Calculator.complete_t_vector(setting, normalized_t_rhs_free, boundary_t_vector)
        return t_vector

    @staticmethod
    def clean_acceleration(scene: Scene, normalized_acceleration):
        if normalized_acceleration is None:
            return None
        if not isinstance(scene, SceneRandomized):
            return normalized_acceleration
        return normalized_acceleration + scene.normalized_a_correction

    @staticmethod
    def clean_temperature(scene, temperature):
        _ = scene
        return temperature if temperature is not None else None

    @staticmethod
    def denormalize(setting, normalized_cleaned_a):
        return setting.denormalize_rotate(normalized_cleaned_a)

    @staticmethod
    def complete_a_vector(setting, normalized_rhs_free, a_contact_vector):
        # a_independent_vector = setting.solver_cache.free_x_free_inverted @ (
        #     normalized_rhs_free - (setting.solver_cache.free_x_contact @ a_contact_vector)
        # )
        s1 = normalized_rhs_free - (setting.solver_cache.free_x_contact @ a_contact_vector)
        a_independent_vector = jxh.solve_linear_jax(
            matrix=setting.solver_cache.free_x_free, vector=s1
        )

        normalized_a = np.vstack(
            (
                nph.unstack(a_contact_vector, setting.dimension),
                nph.unstack(a_independent_vector, setting.dimension),
            )
        )
        return nph.stack(normalized_a)

    @staticmethod
    def complete_t_vector(setting: SceneTemperature, normalized_t_rhs_free, t_contact_vector):
        t_independent_vector = setting.solver_cache.temperature_free_x_free_inv @ (
            normalized_t_rhs_free
            - (setting.solver_cache.temperature_free_x_contact @ t_contact_vector)
        )

        return np.vstack((t_contact_vector, t_independent_vector))
