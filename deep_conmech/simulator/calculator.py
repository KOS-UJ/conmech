import time
from argparse import ArgumentError

import numpy as np
import scipy
from scipy import optimize

from deep_conmech.common import config
from conmech.helpers import nph


class Calculator:
    @staticmethod
    def solve_all(setting, initial_vector=None):
        normalized_a = Calculator.solve_normalized(setting, initial_vector)
        normalized_cleaned_a = Calculator.clean(setting, normalized_a)
        cleaned_a = Calculator.denormalize(setting, normalized_cleaned_a)
        return cleaned_a, normalized_cleaned_a

    @staticmethod
    def solve(setting, initial_vector=None):
        cleaned_a, _ = Calculator.solve_all(setting, initial_vector)
        return cleaned_a

    @staticmethod
    def mode():
        return config.CALCULATOR_MODE

    @staticmethod
    def is_fast():
        return config.CALCULATOR_MODE == "function"

    @staticmethod
    def solve_normalized(setting, initial_vector=None):
        if config.CALCULATOR_MODE == "function":
            return Calculator.solve_normalized_function(setting)
        elif config.CALCULATOR_MODE == "optimization":
            return Calculator.solve_normalized_optimization(setting, initial_vector)
        else:
            raise ArgumentError

    @staticmethod
    def solve_normalized_function(setting):
        normalized_a_vector = np.linalg.solve(setting.C, setting.normalized_E)
        # print(f"Quality: {np.sum(np.mean(C@v_vector-E))}")
        return nph.unstack(normalized_a_vector)

        """
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
    def solve_normalized_optimization(setting, initial_boundary_vector=None):
        if initial_boundary_vector is None:
            initial_boundary_vector = np.zeros(
                setting.boundary_nodes_count * config.DIM
            )

        tstart = time.time()
        normalized_boundary_a_vector_np = scipy.optimize.minimize(
            setting.normalized_L2_obstacle_np,
            initial_boundary_vector,
            method="L-BFGS-B",  # , options={"disp": True}
        ).x
        t_np = time.time() - tstart
        """
        tstart = time.time()
        normalized_boundary_a_vector_nvt = scipy.optimize.minimize(
            setting.normalized_L2_obstacle_nvt,
            initial_boundary_vector,  # , options={"disp": True}
        ).x
        t_nvt = time.time() - tstart
        """
        normalized_boundary_a_vector = normalized_boundary_a_vector_np.reshape(-1, 1)

        normalized_a_vector = Calculator.get_normalized_a_vector(
            setting, setting.normalized_Ei, normalized_boundary_a_vector
        )

        return nph.unstack(normalized_a_vector)

    @staticmethod
    def clean(setting, normalized_a):
        return normalized_a + setting.normalized_a_correction

    @staticmethod
    def denormalize(setting, normalized_cleaned_a):
        return setting.rotate_from_upward(normalized_cleaned_a)

    @staticmethod
    def get_normalized_a_vector(setting, normalized_Ei, normalized_at_vector):
        normalized_ai_vector = setting.CiiINV @ (
            normalized_Ei - (setting.Cit @ normalized_at_vector)
        )

        normalized_a = np.vstack(
            (
                nph.unstack(normalized_at_vector),
                nph.unstack(normalized_ai_vector),
            )
        )
        normalized_a_vector = nph.stack(normalized_a)
        return normalized_a_vector
