from typing import Tuple


class AbstractDynamicsFactory:
    @property
    def dimension(self) -> int:
        raise NotImplementedError()

    def get_edges_features_matrix(self, elements, nodes) -> Tuple:
        raise NotImplementedError()

    def calculate_constitutive_matrices(self, W, mu, lambda_):
        raise NotImplementedError()

    def calculate_acceleration(self, U, density):
        raise NotImplementedError()

    def calculate_thermal_expansion(self, V, coeff):
        raise NotImplementedError()

    def calculate_thermal_conductivity(self, W, coeff):
        raise NotImplementedError()

    def get_piezoelectric_tensor(self, W, coeff):
        raise NotImplementedError()

    def get_permittivity_tensor(self, W, coeff):
        raise NotImplementedError()
