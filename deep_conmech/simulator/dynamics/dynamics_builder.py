import numpy as np

from conmech.dataclass.body_properties import (
    BodyProperties,
    DynamicBodyProperties,
    StaticBodyProperties,
    TemperatureBodyProperties,
)


class DynamicsBuilder:
    def build_matrices(
            self,
            elements: np.ndarray,
            nodes: np.ndarray,
            independent_indices: slice,
            body_prop: StaticBodyProperties,
    ):
        return self.get_matrices(
            elements, nodes, body_prop, independent_indices
        )

    def get_matrices(
            self, elements, nodes, body_prop: BodyProperties, independent_indices
    ):
        edges_features_matrix, element_initial_volume = self.get_edges_features_matrix(elements,
                                                                                       nodes)

        i = independent_indices
        const_volume = edges_features_matrix[0][i, i]
        U = edges_features_matrix[1][i, i]

        ALL_V = [edges_features_matrix[2 + j][i, i] for j in range(self.dimension)]
        ALL_W = [
            edges_features_matrix[2 + self.dimension + j][i, i]
            for j in range(self.dimension ** 2)
        ]

        const_elasticity = (
            self.calculate_constitutive_matrices(
                *ALL_W, body_prop.mu, body_prop.lambda_
            )
            if isinstance(body_prop, StaticBodyProperties)
            else None
        )

        const_viscosity = (
            self.calculate_constitutive_matrices(
                *ALL_W, body_prop.theta, body_prop.zeta
            )
            if isinstance(body_prop, DynamicBodyProperties)
            else None
        )

        ACC = self.calculate_acceleration(U, body_prop.mass_density)

        if isinstance(body_prop, TemperatureBodyProperties):
            C2T = self.calculate_temperature_C(*ALL_V, body_prop.C_coeff)
            K = self.calculate_temperature_K(*ALL_W, body_prop.K_coeff)
        else:
            C2T = None
            K = None

        return element_initial_volume, const_volume, ACC, const_elasticity, const_viscosity, C2T, K

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
