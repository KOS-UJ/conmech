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


class TimeDependentBodyPropertiesTuple(NamedTuple):
    mass_density: float
    mu: float
    lambda_: float
    theta: float
    zeta: float


@dataclass
class TimeDependentBodyProperties(StaticBodyProperties):
    theta: float
    zeta: float

    # 65: TODO: remove
    def get_tuple(self):
        return TimeDependentBodyPropertiesTuple(
            mass_density=self.mass_density,
            mu=self.mu,
            lambda_=self.lambda_,
            theta=self.theta,
            zeta=self.zeta,
        )


@dataclass
class PiezoelectricBodyProperties:
    piezoelectricity: np.ndarray
    permittivity: np.ndarray


@dataclass
class TemperatureBodyProperties:
    thermal_expansion: np.ndarray
    thermal_conductivity: np.ndarray


@dataclass
class StaticTemperatureBodyProperties(StaticBodyProperties, TemperatureBodyProperties):
    pass


@dataclass
class TimeDependentTemperatureBodyProperties(
    TimeDependentBodyProperties,
    TemperatureBodyProperties,
):
    pass


@dataclass
class TimeDependentPiezoelectricBodyProperties(
    TimeDependentBodyProperties,
    PiezoelectricBodyProperties,
):
    pass
