from dataclasses import dataclass

import numpy as np


@dataclass
class BodyProperties:
    mass_density: float


@dataclass
class StaticBodyProperties(BodyProperties):
    mu: float
    lambda_: float


@dataclass
class TimeDependentBodyProperties(StaticBodyProperties):
    theta: float
    zeta: float


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


@dataclass
class BaseFunctionNormBodyProperties(BodyProperties):
    pass
