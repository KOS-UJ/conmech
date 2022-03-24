import time
from argparse import ArgumentError
from typing import Callable, Optional

import numpy as np
import scipy
from conmech.helpers import nph
from deep_conmech.common import config
from deep_conmech.graph.setting.setting_randomized import SettingRandomized
from scipy import optimize


class Calculator:
    @staticmethod
    def solve(
        setting: SettingRandomized, initial_vector: Optional[np.ndarray] = None
    ) -> np.ndarray:
        cleaned_a, _ = Calculator.solve_all(setting, initial_vector)
        return cleaned_a

    @staticmethod
    def solve_all(
        setting: SettingRandomized, initial_vector: Optional[np.ndarray] = None
    ) -> np.ndarray:
        normalized_a = Calculator.solve_normalized(setting, initial_vector)
        normalized_cleaned_a = Calculator.clean(setting, normalized_a)
        cleaned_a = Calculator.denormalize(setting, normalized_cleaned_a)
        return cleaned_a, normalized_cleaned_a

    @staticmethod
    def solve_normalized(
        setting: SettingRandomized, initial_a_vector: Optional[np.ndarray] = None
    ) -> np.ndarray:
        # TODO: repeat with optimization if collision in this round
        if setting.is_coliding:
            return Calculator.solve_normalized_optimization(setting, initial_a_vector)
        else:
            return Calculator.solve_normalized_function(setting, initial_a_vector)

    @staticmethod
    def solve_with_temperature(
        setting: SettingRandomized,
        initial_a_vector: Optional[np.ndarray] = None,
        initial_t_vector: Optional[np.ndarray] = None,
    ) -> np.ndarray:
        cleaned_a = Calculator.solve(setting, initial_a_vector)

        t = Calculator.solve_temperature_normalized(setting, initial_t_vector)
        cleaned_t = Calculator.clean_temperature(setting, t)

        return cleaned_a, cleaned_t

    @staticmethod
    def solve_temperature_normalized(
        setting: SettingRandomized, initial_t_vector: Optional[np.ndarray] = None
    ) -> np.ndarray:
        return Calculator.solve_temperature_normalized_function(
            setting, initial_t_vector
        )

    @staticmethod
    def solve_temperature_normalized_function(setting, initial_t_vector):
        t_vector = np.linalg.solve(setting.T, setting.Q)
        return t_vector

    @staticmethod
    def solve_normalized_function(setting, initial_a_vector):
        normalized_a_vector = np.linalg.solve(setting.C, setting.normalized_E)
        # print(f"Quality: {np.sum(np.mean(C@v_vector-E))}")
        return nph.unstack(normalized_a_vector, setting.dimension)

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
            method="L-BFGS-B",  # , POWELL L-BFGS-B options={"disp": True}
        ).x

    @staticmethod
    def solve_normalized_optimization(setting, initial_boundary_a=None):
        if initial_boundary_a is None:
            initial_boundary_vector = np.zeros(
                setting.boundary_nodes_count * setting.dim
            )
        else:
            initial_boundary_vector = nph.stack_column(
                initial_boundary_a[setting.boundary_indices]
            )

        tstart = time.time()
        cost_function = setting.get_normalized_L2_obstacle_np()
        normalized_boundary_a_vector_np = Calculator.minimize(
            cost_function, initial_boundary_vector
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
        normalized_a_vector = Calculator.get_normalized_a_vector(
            setting, setting.normalized_E_free, normalized_boundary_a_vector
        )

        return nph.unstack(normalized_a_vector, setting.dimension)

    @staticmethod
    def clean(setting, normalized_a):
        return (
            normalized_a + setting.normalized_a_correction
            if normalized_a is not None
            else None
        )

    @staticmethod
    def clean_temperature(setting, t):
        return t if t is not None else None  # + setting.normalized_a_correction #TODO

    @staticmethod
    def denormalize(setting, normalized_cleaned_a):
        return setting.denormalize_rotate(normalized_cleaned_a)

    @staticmethod
    def get_normalized_a_vector(setting, normalized_E_free, normalized_at_vector):
        normalized_ai_vector = setting.free_x_free_inverted @ (
            normalized_E_free - (setting.free_x_contact @ normalized_at_vector)
        )

        normalized_a = np.vstack(
            (
                nph.unstack(normalized_at_vector, setting.dimension),
                nph.unstack(normalized_ai_vector, setting.dimension),
            )
        )
        normalized_a_vector = nph.stack(normalized_a)
        return normalized_a_vector
