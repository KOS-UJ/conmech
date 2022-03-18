import time
from argparse import ArgumentError

import numpy as np
import scipy
from conmech.helpers import nph
from deep_conmech.common import config
from scipy import optimize


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
        return nph.unstack(normalized_a_vector, setting.dim)

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
    def minimize(function, initial_vector):
        return scipy.optimize.minimize(
            function, initial_vector, method="POWELL" #, options={"disp": True}
        ).x

    @staticmethod
    def solve_normalized_optimization(setting, initial_boundary_vector=None):
        if initial_boundary_vector is None:
            initial_boundary_vector = np.zeros(
                setting.boundary_nodes_count * setting.dim
            )
        
        tstart = time.time()
        cost_function = setting.get_normalized_L2_obstacle_np()
        normalized_boundary_a_vector_np = Calculator.minimize(
            cost_function, initial_boundary_vector
        )
        t_np = time.time() - tstart
        '''
        tstart = time.time()
        normalized_boundary_a_vector_nvt = Calculator.minimize(
            setting.normalized_L2_obstacle_nvt, initial_boundary_vector
        ) 
        t_nvt = time.time() - tstart
        '''

        normalized_boundary_a_vector = normalized_boundary_a_vector_np.reshape(-1, 1)
        normalized_a_vector = Calculator.get_normalized_a_vector(
            setting, setting.normalized_Ei, normalized_boundary_a_vector
        )

        return nph.unstack(normalized_a_vector, setting.dim)

    @staticmethod
    def clean(setting, normalized_a):
        return (
            normalized_a + setting.normalized_a_correction
            if normalized_a is not None
            else None
        )

    @staticmethod
    def denormalize(setting, normalized_cleaned_a):
        return setting.denormalize_rotate(normalized_cleaned_a)

    @staticmethod
    def get_normalized_a_vector(setting, normalized_Ei, normalized_at_vector):
        normalized_ai_vector = setting.CiiINV @ (
            normalized_Ei - (setting.Cit @ normalized_at_vector)
        )

        normalized_a = np.vstack(
            (
                nph.unstack(normalized_at_vector, setting.dim),
                nph.unstack(normalized_ai_vector, setting.dim),
            )
        )
        normalized_a_vector = nph.stack(normalized_a)
        return normalized_a_vector
