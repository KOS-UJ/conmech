from typing import Tuple
import numpy as np


class AbstractDynamicsFactory:
    @property
    def dimension(self) -> int:
        raise NotImplementedError()

    def get_edges_features_matrix(self, elements, nodes) -> Tuple:
        raise NotImplementedError()

    def calculate_constitutive_matrices(
        self, W: np.ndarray, mu: float, lambda_: float
    ) -> np.ndarray:
        raise NotImplementedError()

    def calculate_acceleration(self, U: np.ndarray, density: float) -> np.ndarray:
        raise NotImplementedError()

    def calculate_thermal_expansion(self, V: np.ndarray, coeff: np.ndarray) -> np.ndarray:
        raise NotImplementedError()

    def calculate_thermal_conductivity(self, W: np.ndarray, coeff: np.ndarray) -> np.ndarray:
        raise NotImplementedError()

    def get_piezoelectric_tensor(self, W: np.ndarray, coeff: np.ndarray) -> np.ndarray:
        raise NotImplementedError()

    def get_permittivity_tensor(self, W: np.ndarray, coeff: np.ndarray) -> np.ndarray:
        raise NotImplementedError()

    @staticmethod
    def calculate_poisson_matrix(W: np.ndarray) -> np.ndarray:
        return np.sum(W.diagonal(), axis=2)
