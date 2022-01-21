import numpy as np
import scipy
import torch

import config
import helpers


class Calculator:
    def __init__(self, setting, print_raport=None):
        self.setting = setting
        self.print_raport = print_raport

    def function(self, C, E):
        normalized_a_vector = np.linalg.solve(C, E)
        # print(f"Quality: {np.sum(np.mean(C@v_vector-E))}")
        normalized_a = helpers.unstack(normalized_a_vector)
        return normalized_a

    def solve_all_function(self):
        normalized_E = self.setting.get_E_np(
            self.setting.normalized_forces,
            self.setting.normalized_u_old,
            self.setting.normalized_v_old,
        )
        normalized_a = self.function(self.setting.C, normalized_E)
        normalized_cleaned_a = normalized_a + self.setting.normalized_a_correction
        cleaned_a = self.setting.rotate_from_upward(normalized_cleaned_a)
        return cleaned_a, normalized_cleaned_a

    def solve_function(self):
        cleaned_a, _ = self.solve_all_function()
        return cleaned_a
        

    def solve_optimizer(self):
        initial_vector = np.zeros_like(self.setting.initial_points)
        _function = lambda cleaned_normalized_a_vector: self.setting.L2_np(
            helpers.unstack(cleaned_normalized_a_vector)
        )

        cleaned_normalized_a_vector = scipy.optimize.minimize(
            _function, initial_vector, options={"disp": True}
        ).x
        cleaned_normalized_a = helpers.unstack(cleaned_normalized_a_vector)

        cleaned_a = self.setting.rotate_from_upward(cleaned_normalized_a)
        return cleaned_a


