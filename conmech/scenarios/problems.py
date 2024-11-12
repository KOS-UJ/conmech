"""
Contact Mechanics Problem setups
"""

from abc import ABC
from dataclasses import dataclass
from typing import Optional, Callable

import numpy as np

from conmech.dynamics.contact.contact_law import ContactLaw
from conmech.dynamics.contact.interior_contact_law import InteriorContactLaw
from conmech.mesh.boundaries_description import BoundariesDescription
from conmech.properties.mesh_description import MeshDescription

from conmech.dynamics.statement import (
    Statement,
    StaticDisplacementStatement,
    QuasistaticVelocityStatement,
    DynamicVelocityWithTemperatureStatement,
    TemperatureStatement,
    PiezoelectricStatement,
    DynamicVelocityStatement,
    QuasistaticVelocityWithPiezoelectricStatement,
    QuasistaticRelaxationStatement,
    StaticPoissonStatement,
    WaveStatement,
)

# pylint: disable=too-many-ancestors


@dataclass
class Problem(ABC):
    # pylint: disable=unused-argument
    mesh_descr: MeshDescription
    boundaries: BoundariesDescription

    @classmethod
    def statement(cls, body) -> Optional[Statement]:
        raise NotImplementedError()

    @classmethod
    def second_statement(cls, body) -> Optional[Statement]:
        return None

    @staticmethod
    def inner_forces(
        x: np.ndarray, *, v: Optional[np.ndarray] = None, t: Optional[float] = None
    ) -> np.ndarray:
        return np.zeros_like(x)

    @staticmethod
    def outer_forces(
        x: np.ndarray, *, v: Optional[np.ndarray] = None, t: Optional[float] = None
    ) -> np.ndarray:
        return np.zeros_like(x)


class StaticProblem(Problem, ABC):
    pass


class TimeDependentProblem(Problem, ABC):
    time_step = 0.0


class QuasistaticProblem(TimeDependentProblem, ABC):
    pass


class DynamicProblem(TimeDependentProblem, ABC):
    pass


@dataclass
class PoissonProblem(StaticProblem, ABC):
    # pylint: disable=unused-argument
    @classmethod
    def statement(cls, body) -> Optional[Statement]:
        return None

    @classmethod
    def second_statement(cls, body) -> Optional[Statement]:
        return StaticPoissonStatement(body)

    @staticmethod
    def initial_temperature(x: np.ndarray) -> np.ndarray:
        return np.zeros(len(x))

    @staticmethod
    def internal_temperature(x: np.ndarray, t: Optional[float] = None) -> np.ndarray:
        return np.zeros(len(x))

    @staticmethod
    def outer_temperature(x: np.ndarray, t: Optional[float] = None) -> np.ndarray:
        return np.zeros(len(x))


@dataclass
class WaveProblem(DynamicProblem, ABC):
    propagation: float

    @classmethod
    def statement(cls, body) -> Statement:
        return WaveStatement(body)

    @staticmethod
    def initial_displacement(x: np.ndarray) -> np.ndarray:
        return np.zeros_like(x)

    @staticmethod
    def initial_velocity(x: np.ndarray) -> np.ndarray:
        return np.zeros_like(x)


@dataclass
class ContactWaveProblem(WaveProblem, ABC):
    contact_law: ContactLaw


@dataclass
class InteriorContactWaveProblem(ContactWaveProblem, ABC):
    contact_law: InteriorContactLaw


@dataclass
class DisplacementProblem(Problem, ABC):
    mu_coef: float
    la_coef: float
    contact_law: ContactLaw
    dynamism: str = None  # TODO: #65 remove

    @staticmethod
    def initial_displacement(x: np.ndarray) -> np.ndarray:
        return np.zeros_like(x)


class StaticDisplacementProblem(DisplacementProblem, StaticProblem, ABC):
    @classmethod
    def statement(cls, body) -> Statement:
        return StaticDisplacementStatement(body)


class TimeDependentDisplacementProblem(DisplacementProblem, ABC):
    th_coef: float
    ze_coef: float
    time_step: float

    @staticmethod
    def initial_velocity(x: np.ndarray) -> np.ndarray:
        return np.zeros_like(x)


class QuasistaticDisplacementProblem(QuasistaticProblem, TimeDependentDisplacementProblem, ABC):
    @classmethod
    def statement(cls, body) -> Statement:
        return QuasistaticVelocityStatement(body)


class DynamicDisplacementProblem(DynamicProblem, TimeDependentDisplacementProblem, ABC):
    @classmethod
    def statement(cls, body) -> Statement:
        return DynamicVelocityStatement(body)


class TemperatureTimeDependentProblem(TimeDependentDisplacementProblem, ABC):
    contact_law_2: ContactLaw
    thermal_expansion: np.ndarray
    thermal_conductivity: np.ndarray

    @classmethod
    def second_statement(cls, body) -> Optional[Statement]:
        return TemperatureStatement(body)

    @staticmethod
    def initial_temperature(x: np.ndarray) -> np.ndarray:
        return np.zeros(len(x))


class PiezoelectricTimeDependentProblem(TimeDependentDisplacementProblem, ABC):
    contact_law_2: ContactLaw
    piezoelectricity: np.ndarray
    permittivity: np.ndarray

    @classmethod
    def second_statement(cls, body) -> Optional[Statement]:
        return PiezoelectricStatement(body)

    @staticmethod
    def initial_electric_potential(x: np.ndarray) -> np.ndarray:
        return np.zeros(len(x))


class RelaxationQuasistaticProblem(QuasistaticDisplacementProblem, ABC):
    relaxation: Callable[[float], np.ndarray]

    @classmethod
    def statement(cls, body) -> Statement:
        return QuasistaticRelaxationStatement(body)

    @staticmethod
    def initial_absement(x: np.ndarray) -> np.ndarray:
        return np.zeros_like(x)


class TemperatureDynamicProblem(DynamicProblem, TemperatureTimeDependentProblem, ABC):
    @classmethod
    def statement(cls, body) -> Statement:
        return DynamicVelocityWithTemperatureStatement(body)


class PiezoelectricQuasistaticProblem(
    QuasistaticDisplacementProblem, PiezoelectricTimeDependentProblem, ABC
):
    @classmethod
    def statement(cls, body) -> Statement:
        return QuasistaticVelocityWithPiezoelectricStatement(body)


class PiezoelectricDynamicProblem(
    DynamicDisplacementProblem, PiezoelectricTimeDependentProblem, ABC
):
    pass
