from dataclasses import dataclass
from typing import Callable

import numpy as np


@dataclass
class BodyProperties:
    mass_density: float


@dataclass
class ElasticProperties(BodyProperties):
    mu: float
    lambda_: float


@dataclass
class ViscoelasticProperties(ElasticProperties):
    theta: float
    zeta: float


@dataclass
class RelaxationBodyProperties:
    relaxation: Callable[[float], np.ndarray]


@dataclass
class PiezoelectricBodyProperties:
    piezoelectricity: np.ndarray
    permittivity: np.ndarray


@dataclass
class TemperatureBodyProperties:
    thermal_expansion: np.ndarray
    thermal_conductivity: np.ndarray


@dataclass
class ElasticTemperatureProperties(ElasticProperties, TemperatureBodyProperties):
    pass


@dataclass
class ViscoelasticTemperatureProperties(
    ViscoelasticProperties,
    TemperatureBodyProperties,
):
    pass


@dataclass
class ViscoelasticPiezoelectricProperties(
    ViscoelasticProperties,
    PiezoelectricBodyProperties,
):
    pass


@dataclass
class ElasticRelaxationProperties(ElasticProperties, RelaxationBodyProperties):
    pass
