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
        self.temperature = BodyForces(
            body,
        )
        self.factory = get_factory(body.mesh.mesh_prop.dimension)
        self.element_initial_volume: np.ndarray
        self.volume_at_nodes: np.ndarray
        self.acceleration_operator: np.ndarray
        self.elasticity: np.ndarray
        self.viscosity: np.ndarray
        self._w_matrix = None
        self.__relaxation: Optional[np.ndarray] = None
        self.__relaxation_tensor: Optional[float] = None
        self.thermal_expansion: np.ndarray
        self.thermal_conductivity: np.ndarray
        self.piezoelectricity: np.ndarray
        self.permittivity: np.ndarray
        self.poisson_operator: np.ndarray

        self.reinitialize_matrices()

    def reinitialize_matrices(self):
        (
            self.element_initial_volume,
            self.volume_at_nodes,
            U,
            V,
            self._w_matrix,
        ) = get_basic_matrices(
            elements=self.body.mesh.elements, nodes=self.body.mesh.initial_nodes
        )  # + self.displacement_old)
        (
            self.acceleration_operator,
            self.elasticity,
            self.viscosity,
            self.thermal_expansion,
            self.thermal_conductivity,
            self.piezoelectricity,
            self.permittivity,
            self.poisson_operator,
        ) = get_dynamics(
            elements=self.body.mesh.elements,
            body_prop=self.body.properties,
            U=U,
            V=V,
            W=self._w_matrix,
        )

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
