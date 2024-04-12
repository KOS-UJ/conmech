from typing import Optional

import numpy as np

from conmech.dynamics.factory.dynamics_factory_method import (
    get_dynamics,
    get_basic_matrices,
    get_factory,
)
from conmech.properties.body_properties import (
    ElasticRelaxationProperties,
)
from conmech.scene.body_forces import BodyForces


class Dynamics:
    # pylint: disable=too-many-instance-attributes
    def __init__(
        self,
        body: "Body",
    ):
        self.body = body
        self.body.dynamics = self

        self.force = BodyForces(body)
        self.temperature = BodyForces(body)

        self.factory = get_factory(body.mesh.dimension)
        self.element_initial_volume: np.ndarray
        self.volume_at_nodes: np.ndarray
        self.acceleration_operator: np.ndarray
        self.elasticity: np.ndarray
        self.viscosity: np.ndarray
        self._w_matrix: Optional[np.ndarray] = None
        self._local_stifness_matrices: Optional[np.ndarray] = None
        self.__relaxation: Optional[np.ndarray] = None
        self.__relaxation_tensor: Optional[float] = None
        self.thermal_expansion: np.ndarray
        self.thermal_conductivity: np.ndarray
        self.piezoelectricity: np.ndarray
        self.permittivity: np.ndarray
        self.poisson_operator: np.ndarray

        self.reinitialize_matrices()

    def reinitialize_matrices(self, elements_density: Optional[np.ndarray] = None):
        (
            self.element_initial_volume,
            self.volume_at_nodes,
            U,
            V,
            self._w_matrix,
            self._local_stifness_matrices,
        ) = get_basic_matrices(
            elements=self.body.mesh.elements, nodes=self.body.mesh.nodes
        )  # + self.displacement_old)

        if elements_density is not None:
            self._w_matrix = self.asembly_w_matrix_with_density(elements_density)

        (
            self.acceleration_operator,
            self.elasticity,
            self.viscosity,
            self.thermal_expansion,
            self.thermal_conductivity,
            self.piezoelectricity,
            self.permittivity,
            self.poisson_operator,
            self.wave_operator,
        ) = get_dynamics(
            elements=self.body.mesh.elements,
            body_prop=self.body.properties,
            U=U,
            V=V,
            W=self._w_matrix,
        )

    def asembly_w_matrix_with_density(self, elements_density: np.ndarray):
        w_matrix = np.zeros_like(self._w_matrix)
        for element_index, element in enumerate(self.body.mesh.elements):
            for i, global_i in enumerate(element):
                for j, global_j in enumerate(element):
                    w_matrix[:, :, global_i, global_j] += (
                        elements_density[element_index]
                        * self._local_stifness_matrices[:, :, element_index, i, j]
                    )
        return w_matrix

    def relaxation(self, time: float = 0):
        # TODO handle others
        if isinstance(self.body.properties, ElasticRelaxationProperties):
            relaxation_tensor = self.body.properties.relaxation(time)
            if (relaxation_tensor != self.__relaxation_tensor).any():
                self.__relaxation_tensor = relaxation_tensor
                self.__relaxation = self.factory.get_relaxation_tensor(
                    self._w_matrix, relaxation_tensor
                )
        else:
            raise TypeError("There is no relaxation operator!")

        return self.__relaxation
