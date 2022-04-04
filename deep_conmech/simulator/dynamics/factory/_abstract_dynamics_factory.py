from typing import Tuple


class AbstractDynamicsFactory:
    @property
    def dimension(self) -> int:
        raise NotImplementedError()

    def get_edges_features_matrix(self, elements, nodes) -> Tuple:
        raise NotImplementedError()

    def calculate_constitutive_matrices(self, W, MU, LA):
        raise NotImplementedError()

    def calculate_acceleration(self, U, density):
        raise NotImplementedError()

    def calculate_temperature_C(self, V, C_coef):
        raise NotImplementedError()

    def calculate_temperature_K(self, W, K_coef):
        raise NotImplementedError()
