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
class DynamicBodyProperties(StaticBodyProperties):
    theta: float
    zeta: float


@dataclass
class PiezoelectricityBodyProperties:
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
class DynamicTemperatureBodyProperties(DynamicBodyProperties, TemperatureBodyProperties):
    pass
