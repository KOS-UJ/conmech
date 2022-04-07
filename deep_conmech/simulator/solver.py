import time
from argparse import ArgumentError
from typing import Callable, Optional

import numpy as np
import scipy
from scipy import optimize

from conmech.helpers import nph
from deep_conmech.graph.setting.setting_randomized import SettingRandomized
from deep_conmech.simulator.setting.setting_iterable import SettingIterable
from deep_conmech.simulator.setting.setting_temperature import \
    SettingTemperature


class Solver:
    """
        time used
        base (BFGS) - 178 / 1854
        Nelder-Mead - 883
        CG - 96 / 1458.23
        POWELL - 313
        Newton-CG - n/a
        L-BFGS-B - 23 / 191
        TNC - 672
        COBYLA - 298
        SLSQP - 32 / 210 - bad transfer
        trust-constr - 109
        dogleg - n/a
        trust-ncg - n/a
        trust-exact - n/a
        trust-krylov - n/a
    """

    @staticmethod
    def minimize(
            function: Callable[[np.ndarray], np.ndarray], initial_vector: np.ndarray
    ) -> np.ndarray:
        return scipy.optimize.minimize(
            function,
            initial_vector,
            method="L-BFGS-B",  # POWELL, L-BFGS-B, options={"disp": True}
        ).x

    @staticmethod
    def solve(
            setting: SettingRandomized, initial_a: Optional[np.ndarray] = None
    ) -> np.ndarray:
        cleaned_a, _ = Solver.solve_all(setting, initial_a)
        return cleaned_a

    @staticmethod
    def solve_all(setting: SettingIterable, initial_a: Optional[np.ndarray] = None):
        normalized_a = Solver.solve_acceleration_normalized(setting, None, initial_a)
        normalized_cleaned_a = Solver.clean_acceleration(setting, normalized_a)
        cleaned_a = Solver.denormalize(setting, normalized_cleaned_a)
        return cleaned_a, normalized_cleaned_a

    @staticmethod
    def solve_with_temperature(
            setting: SettingTemperature,
            initial_a: Optional[np.ndarray] = None,
            initial_t: Optional[np.ndarray] = None,
    ):
        uzawa = False
        max_iter = 10
        i = 0
        normalized_a = None
        t = setting.t_old
        while i < 2 or np.allclose(last_normalized_a, normalized_a) == False and np.allclose(last_t,
                                                                                             t) == False:
            last_normalized_a, last_t = normalized_a, t
            normalized_a = Solver.solve_acceleration_normalized(setting, t, initial_a)
            t = Solver.solve_temperature_normalized(setting, normalized_a, initial_t)
            i += 1
            if i >= max_iter:
                raise ArgumentError(f"Uzawa algorithm: maximum of {max_iter} iterations exceeded")
            if uzawa is False:
                break

        normalized_cleaned_a = Solver.clean_acceleration(setting, normalized_a)
        cleaned_a = Solver.denormalize(setting, normalized_cleaned_a)
        cleaned_t = Solver.clean_temperature(setting, t)
        return cleaned_a, cleaned_t

    @staticmethod
    def solve_temperature(setting: SettingTemperature, normalized_a: np.ndarray, initial_t):
        t = Solver.solve_temperature_normalized(setting, normalized_a, initial_t)
        cleaned_t = Solver.clean_temperature(setting, t)
        return cleaned_t

    @staticmethod
    def solve_acceleration_normalized(
            setting: SettingIterable, t, initial_a: Optional[np.ndarray] = None
    ) -> np.ndarray:
        # TODO: #62 repeat with optimization if collision in this round
        if setting.is_colliding:
            return Solver.solve_acceleration_normalized_optimization(setting, t, initial_a)
        else:
            return Solver.solve_acceleration_normalized_function(setting, t, initial_a)

    @staticmethod
    def solve_temperature_normalized(
            setting: SettingTemperature, normalized_a: np.ndarray,
            initial_t: Optional[np.ndarray] = None
    ) -> np.ndarray:
        # TODO: #62 repeat with optimization if collision in this round
        if setting.is_colliding:
            return Solver.solve_temperature_normalized_optimization(setting, normalized_a,
                                                                    initial_t)
        else:
            return Solver.solve_temperature_normalized_function(setting, normalized_a, initial_t)

    @staticmethod
    def solve_temperature_normalized_function(setting: SettingTemperature, normalized_a: np.ndarray,
                                              initial_t):
        normalized_Q = setting.get_normalized_Q_np(normalized_a)
        t_vector = np.linalg.solve(setting.T, normalized_Q)
        return t_vector

    @staticmethod
    def solve_acceleration_normalized_function(setting, t, initial_a=None):
        normalized_E = setting.get_normalized_E_np(t)
        normalized_a_vector = np.linalg.solve(setting.C, normalized_E)
        # print(f"Quality: {np.sum(np.mean(C@v_vector-E))}")
        return nph.unstack(normalized_a_vector, setting.dimension)

    @staticmethod
    def solve_acceleration_normalized_optimization(setting, t, initial_a=None):
        if initial_a is None:
            initial_a_boundary_vector = np.zeros(
                setting.boundary_nodes_count * setting.dimension
            )
        else:
            initial_a_boundary_vector = nph.stack_column(
                initial_a[setting.boundary_indices]
            )

        tstart = time.time()
        cost_function, normalized_E_free = setting.get_normalized_L2_obstacle_np(t)
        normalized_boundary_a_vector_np = Solver.minimize(
            cost_function, initial_a_boundary_vector
        )
        t_np = time.time() - tstart
        """
        tstart = time.time()
        normalized_boundary_a_vector_nvt = Calculator.minimize(
            setting.normalized_L2_obstacle_nvt, initial_boundary_vector
        ) 
        t_nvt = time.time() - tstart
        """

        normalized_boundary_a_vector = normalized_boundary_a_vector_np.reshape(-1, 1)
        normalized_a_vector = Solver.complete_a_vector(
            setting, normalized_E_free, normalized_boundary_a_vector
        )

        return nph.unstack(normalized_a_vector, setting.dimension)

    @staticmethod
    def solve_temperature_normalized_optimization(
            setting: SettingTemperature,
            normalized_a: np.ndarray,
            initial_t_vector: np.ndarray,
    ):
        initial_t_boundary_vector = np.zeros(setting.boundary_nodes_count)

        cost_function, normalized_Q_free = setting.get_normalized_L2_temperature_np(normalized_a)
        boundary_t_vector_np = Solver.minimize(cost_function, initial_t_boundary_vector)

        boundary_t_vector = boundary_t_vector_np.reshape(-1, 1)
        t_vector = Solver.complete_t_vector(
            setting, normalized_Q_free, boundary_t_vector
        )
        return t_vector

    @staticmethod
    def clean_acceleration(setting: SettingIterable, normalized_a):
        if normalized_a is None:
            return None
        if isinstance(setting, SettingRandomized) == False:
            return normalized_a
        return normalized_a + setting.normalized_a_correction

    @staticmethod
    def clean_temperature(setting, t):
        return t if t is not None else None  # + setting.normalized_a_correction #TODO

    @staticmethod
    def denormalize(setting, normalized_cleaned_a):
        return setting.denormalize_rotate(normalized_cleaned_a)

    @staticmethod
    def complete_a_vector(setting, normalized_E_free, a_contact_vector):
        a_independent_vector = setting.free_x_free_inverted @ (
                normalized_E_free - (setting.free_x_contact @ a_contact_vector)
        )

        normalized_a = np.vstack(
            (
                nph.unstack(a_contact_vector, setting.dimension),
                nph.unstack(a_independent_vector, setting.dimension),
            )
        )
        return nph.stack(normalized_a)

    @staticmethod
    def complete_t_vector(
            setting: SettingTemperature, normalized_Q_free, t_contact_vector
    ):
        t_independent_vector = setting.T_free_x_free_inverted @ (
                normalized_Q_free - (setting.T_free_x_contact @ t_contact_vector)
        )

        return np.vstack((t_contact_vector, t_independent_vector))
