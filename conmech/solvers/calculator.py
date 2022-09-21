from ctypes import ArgumentError
from typing import Optional

import jax
import jax.experimental
import jax.numpy as jnp
import jax.scipy
import jax.scipy.optimize
import numpy as np

from conmech.helpers import cmh, jxh, nph
from conmech.scene.body_forces import energy
from conmech.scene.scene import (
    Scene,
    energy_obstacle_colliding_jax,
    energy_obstacle_jax,
)
from conmech.scene.scene_temperature import SceneTemperature
from deep_conmech.scene.scene_randomized import SceneRandomized
from optimization.lbfgs import minimize_lbfgs


class Calculator:

    MAX_K = 0

    @staticmethod
    def minimize_jax(function, initial_vector: np.ndarray, args) -> np.ndarray:
        # result = scipy.optimize.minimize(
        #     function,
        #     initial_vector,
        #     method="L-BFGS-B",
        # )
        # return result.x

        x0 = jnp.asarray(initial_vector)

        state = cmh.profile(
            lambda: minimize_lbfgs(fun=function, args=args, x0=x0),
            baypass=True,
        )
        # return np.array(state.x_k)

        # jac = jax.jit(jax.grad(function))
        # result = cmh.profile(
        #     lambda: scipy.optimize.minimize(
        #         function,
        #         x0,
        #         jac=jac,
        #         args=(args,),
        #         method="L-BFGS-B",
        #     ),
        #     baypass=True,
        # )
        # return result.x

        # result = cmh.profile(
        #     lambda: jax.scipy.optimize.minimize(
        #         function,
        #         x0,
        #         args=(args,),
        #         method="l-bfgs-experimental-do-not-rely-on-this",
        #     ),
        #     baypass=False,
        # )
        # return np.array(result.x)

        # hvp = lambda f, x, v: jax.grad(lambda x: jnp.vdot(jax.grad(f)(x, args), v))(x)
        # hes_jax = jax.jit(lambda x: hvp(function, x, x))
        # hes_at_x0 = hes_jax(x0)  # jnp.zeros_like(x0)
        # q, _ = jax.scipy.sparse.linalg.cg(A=hes_jax, b=x0)

        # state = cmh.profile(
        #     lambda: minimize_lbfgs(
        #         fun=function,
        #         args=args,
        #         x0=x0,
        #         hes=None,  # hes,  # None,
        #         xtol_max=0.1,  # 0.1 * scale,
        #         xtol_mean=0.1,  # 0.001 * scale,
        #         max_iter=500,
        #     ),
        #     baypass=True,
        # )

        if Calculator.MAX_K < state.k:
            Calculator.MAX_K = state.k
        # if state.failed:
        #    print("Optimization failed")
        # if state.overrun:
        #     print(
        # f"Optimization overrun: xdiff_max: {state.xdiff_max}, xdiff_mean: {state.xdiff_mean}"
        #     )
        return np.asarray(state.x_k)

        # hes_jax = jax.hessian(function)
        # hes = hes_jax(x0)  # jnp.zeros_like(x0)

        # hvp = lambda f, x, v: jax.grad(lambda x: jnp.vdot(jax.grad(f)(x), v))(x)
        # hes = jax.jit(lambda x: hvp(function, x, x))

        # jac_jax = jax.jit(jax.jacfwd(function)).lower(x0).compile()
        # jacobian_fast = lambda x: np.array(jac_jax(jnp.asarray(x)), dtype=np.float64)

        # hes_jax = jax.jit(jax.hessian(function)).lower(x0).compile()
        # hessian_fast = lambda x: np.array(hes_jax(jnp.asarray(x)), dtype=np.float64)

        # jax.scipy.optimize.minimize(

        # result = cmh.profile(
        #     lambda: scipy.optimize.minimize(
        #         function_fast,
        #         initial_vector,
        #         method="L-BFGS-B",
        #         jac=jacobian_fast,
        #         # hess=hessian_fast,
        #         options={"disp": False},
        #     ),
        #     baypass=False,
        # )
        # return np.asarray(result.x)

    @staticmethod
    def solve(scene: Scene, initial_a: Optional[np.ndarray] = None) -> np.ndarray:
        normalized_a = Calculator.solve_acceleration_normalized(scene, initial_a=initial_a)
        normalized_cleaned_a = Calculator.clean_acceleration(scene, normalized_a)
        cleaned_a = Calculator.denormalize(scene, normalized_cleaned_a)
        return cleaned_a

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
        scene: Scene, temperature=None, initial_a: Optional[np.ndarray] = None
    ) -> np.ndarray:
        # TODO: #62 repeat with optimization if collision in this round
        if True:  # setting.is_colliding():
            return Calculator.solve_acceleration_normalized_optimization_jax(
                scene, temperature=temperature, initial_a=initial_a
            )
        return Calculator.solve_acceleration_normalized_function(
            scene=scene, temperature=temperature, initial_a=initial_a
        )

    @staticmethod
    def solve_temperature_normalized(
        scene: SceneTemperature,
        normalized_acceleration: np.ndarray,
        initial_t: Optional[np.ndarray] = None,
    ) -> np.ndarray:
        # TODO: #62 repeat with optimization if collision in this round
        if True:  # setting.is_colliding():
            return Calculator.solve_temperature_normalized_optimization(
                scene, normalized_acceleration, initial_t
            )
        return Calculator.solve_temperature_normalized_function(
            scene, normalized_acceleration, initial_t
        )

    @staticmethod
    def solve_temperature_normalized_function(
        scene: SceneTemperature, normalized_acceleration: np.ndarray, initial_t
    ):
        _ = initial_t
        normalized_rhs = scene.get_normalized_t_rhs_jax(normalized_acceleration)
        t_vector = np.linalg.solve(scene.solver_cache.lhs_temperature, normalized_rhs)
        return t_vector

    @staticmethod
    def solve_acceleration_normalized_function(scene, temperature=None, initial_a=None):
        # normalized_a_vector, _ = scipy.sparse.linalg.cg(A=A, b=b)

        matrix = scene.solver_cache.lhs_sparse_jax
        vector = scene.get_normalized_rhs_jax(temperature)
        initial_point = jnp.array(nph.stack_column(initial_a)) if initial_a is not None else None

        preconditioner = scene.solver_cache.lhs_preconditioner_jax

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

        normalized_cleaned_a = Calculator.clean_acceleration(scene, normalized_a)
        cleaned_a = Calculator.denormalize(scene, normalized_cleaned_a)
        return cleaned_a

    @staticmethod
    def solve_acceleration_normalized_optimization_jax(scene, temperature=None, initial_a=None):
        if initial_a is None:
            initial_a_vector = np.zeros(scene.nodes_count * scene.dimension)
        else:
            initial_a_vector = nph.stack(initial_a)

        args = cmh.profile(
            lambda: scene.get_energy_obstacle_args_for_jax(temperature),
            baypass=True,
        )

        def get_vector():
            if not scene.is_colliding():
                return Calculator.minimize_jax(
                    function=energy_obstacle_jax,
                    initial_vector=initial_a_vector,
                    args=args,
                )
            return Calculator.minimize_jax(
                function=energy_obstacle_colliding_jax,
                initial_vector=initial_a_vector,
                args=args,
            )

        normalized_a_vector_np = cmh.profile(
            get_vector,
            baypass=True,  # False,
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

        normalized_t_vector = Calculator.minimize_jax(cost_function, initial_t_vector, None)
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
        boundary_t_vector_np = Calculator.minimize_jax(cost_function, initial_t_boundary_vector)

        boundary_t_vector = boundary_t_vector_np.reshape(-1, 1)
        t_vector = Calculator.complete_t_vector(scene, normalized_t_rhs_free, boundary_t_vector)
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
