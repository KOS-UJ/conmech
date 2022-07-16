from collections import namedtuple
from dataclasses import dataclass
from typing import NamedTuple

import numpy as np


@dataclass
class BodyProperties:
    mass_density: float


@dataclass
class StaticBodyProperties(BodyProperties):
    mu: float
    lambda_: float


class DynamicBodyPropertiesTuple(NamedTuple):
    mass_density: float
    mu: float
    lambda_: float
    theta: float
    zeta: float


@dataclass
class DynamicBodyProperties(StaticBodyProperties):
    theta: float
    zeta: float

    def get_tuple(self):
        return DynamicBodyPropertiesTuple(
            mass_density=self.mass_density,
            mu=self.mu,
            lambda_=self.lambda_,
            theta=self.theta,
            zeta=self.zeta,
        )


@dataclass
class TemperatureBodyProperties:
    thermal_expansion: np.ndarray
    thermal_conductivity: np.ndarray


@dataclass
class StaticTemperatureBodyProperties(StaticBodyProperties, TemperatureBodyProperties):
    pass


@dataclass
class DynamicTemperatureBodyProperties(DynamicBodyProperties, TemperatureBodyProperties):
    pass
