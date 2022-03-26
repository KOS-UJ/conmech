from typing import Optional

import numpy as np
from conmech.dataclass.body_properties import (
    BodyProperties, DynamicBodyProperties, DynamicTemperatureBodyProperties, StaticBodyProperties,
    StaticTemperatureBodyProperties, TemperatureBodyProperties)
from numba import njit


class DynamicsBuilder:
    def build_matrices(
        self,
        elements: np.ndarray,
        normalized_points: np.ndarray,
        independent_indices: slice,
        body_prop: StaticBodyProperties,
    ):
        (
            edges_features_matrix,
            element_initial_volume,
        ) = self.get_edges_features_matrix(elements, normalized_points)

        VOL, ACC, A, B, C2T, K = self.get_matrices(
            edges_features_matrix, body_prop, independent_indices
        )

        return element_initial_volume, VOL, ACC, A, B, C2T, K

    @property
    def dimension(self) -> int:
        pass

    def get_edges_features_matrix(self, elements, nodes):
        pass

    def calculate_constitutive_matrices(self, W11, W12, W21, W22, MU, LA):
        pass

    def calculate_acceleration(self, U, density):
        pass

    def calculate_temperature_C(self, V1, V2, C_coef):
        pass

    def calculate_temperature_K(self, W11, W12, W21, W22, K_coef):
        pass

    def get_matrices(
        self, edges_features_matrix, body_prop: DynamicBodyProperties, independent_indices
    ):
        i = independent_indices

        VOL = edges_features_matrix[0][i, i]
        U = edges_features_matrix[1][i, i]

        ALL_V = [edges_features_matrix[2 + j][i, i] for j in range(self.dimension)]
        ALL_W = [
            edges_features_matrix[2 + self.dimension + j][i, i]
            for j in range(self.dimension ** 2)
        ]

        A = self.calculate_constitutive_matrices(
            *ALL_W, body_prop.theta, body_prop.zeta
        )
        B = self.calculate_constitutive_matrices(
            *ALL_W, body_prop.mu, body_prop.lambda_
        )
        ACC = self.calculate_acceleration(U, body_prop.mass_density)

        if isinstance(body_prop, TemperatureBodyProperties):
            C2T = self.calculate_temperature_C(*ALL_V, body_prop.C_coeff)
            K = self.calculate_temperature_K(*ALL_W, body_prop.K_coeff)
        else:
            C2T = None
            K = None

        return VOL, ACC, A, B, C2T, K

