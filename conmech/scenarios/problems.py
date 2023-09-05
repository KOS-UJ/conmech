"""
Contact Mechanics Problem setups
"""
from abc import ABC
from dataclasses import dataclass, field
from typing import Tuple, Union, Optional, Callable

import numpy as np

from conmech.mesh.mesh import MeshProperties
from conmech.mesh.boundaries_description import BoundariesDescription


# pylint: disable=too-many-ancestors


class ContactLaw:
    @staticmethod
    def potential_normal_direction(u_nu: float) -> float:
        raise NotImplementedError()

    @staticmethod
    def subderivative_normal_direction(u_nu: float, v_nu: float) -> float:
        raise NotImplementedError()

    @staticmethod
    def regularized_subderivative_tangential_direction(
        u_tau: np.ndarray, v_tau: np.ndarray, rho=1e-7
    ) -> float:
        """
        Coulomb regularization
        """
        raise NotImplementedError()


@dataclass
class Problem(ABC):
    # pylint: disable=unused-argument
    dimension = 2  # TODO #74 : Not used?
    mesh_type: str
    boundaries: BoundariesDescription
    grid_height: float
    grid_width: float = field(init=False)
    mesh_prop: MeshProperties = field(init=False)

    elements_number: Union[Tuple[int, int], Tuple[int, int, int]]  # number of triangles per aside

    def __post_init__(self):
        self.grid_width = (
            self.grid_height / self.elements_number[0]
        ) * self.elements_number[1]

        self.mesh_prop = MeshProperties(
                dimension=self.dimension,
                mesh_type=self.mesh_type,
                mesh_density=[self.elements_number[1], self.elements_number[0]],
                scale=[float(self.grid_width), float(self.grid_height)],
            )


    @staticmethod   
    def inner_forces(x: np.ndarray, t: Optional[float] = None) -> np.ndarray:
        return np.zeros_like(x)

    @staticmethod
    def outer_forces(x: np.ndarray, t: Optional[float] = None) -> np.ndarray:
        return np.zeros_like(x)


class StaticProblem(Problem, ABC):
    pass


class TimeDependentProblem(Problem, ABC):
    time_step = 0


class QuasistaticProblem(TimeDependentProblem, ABC):
    pass


class DynamicProblem(TimeDependentProblem, ABC):
    pass


@dataclass
class PoissonProblem(StaticProblem, ABC):  # TODO: rename
    # pylint: disable=unused-argument
    @staticmethod
    def initial_temperature(x: np.ndarray) -> np.ndarray:
        return np.zeros_like(len(x))

    @staticmethod
    def internal_temperature(x: np.ndarray, t: Optional[float] = None) -> np.ndarray:
        return np.zeros_like(len(x))

    @staticmethod
    def external_temperature(x: np.ndarray, t: Optional[float] = None) -> np.ndarray:
        return np.zeros_like(len(x))


@dataclass
class DisplacementProblem(Problem, ABC):
    elements_number: Union[Tuple[int, int], Tuple[int, int, int]]  # number of triangles per aside

    mu_coef: float
    la_coef: float
    contact_law: ContactLaw
    dynamism: str = None  # TODO: #65 remove

    @staticmethod
    def initial_displacement(x: np.ndarray) -> np.ndarray:
        return np.zeros_like(x)

    @staticmethod
    def friction_bound(u_nu: float) -> float:
        raise NotImplementedError()


class StaticDisplacementProblem(DisplacementProblem, StaticProblem, ABC):
    @staticmethod
    def friction_bound(u_nu: float) -> float:
        raise NotImplementedError()


class TimeDependentDisplacementProblem(DisplacementProblem, ABC):
    th_coef: float
    ze_coef: float
    time_step: float

    @staticmethod
    def initial_velocity(x: np.ndarray) -> np.ndarray:
        return np.zeros_like(x)


class QuasistaticDisplacementProblem(QuasistaticProblem, TimeDependentDisplacementProblem, ABC):
    pass


class DynamicDisplacementProblem(DynamicProblem, TimeDependentDisplacementProblem, ABC):
    pass


class TemperatureTimeDependentProblem(TimeDependentDisplacementProblem, ABC):
    thermal_expansion: np.ndarray
    thermal_conductivity: np.ndarray

    @staticmethod
    def initial_temperature(x: np.ndarray) -> np.ndarray:
        return np.zeros_like(len(x))


class PiezoelectricTimeDependentProblem(TimeDependentDisplacementProblem, ABC):
    piezoelectricity: np.ndarray
    permittivity: np.ndarray

    @staticmethod
    def initial_electric_potential(x: np.ndarray) -> np.ndarray:
        return np.zeros_like(len(x))


class RelaxationQuasistaticProblem(QuasistaticDisplacementProblem, ABC):
    relaxation: Callable[[float], np.ndarray]

    @staticmethod
    def initial_absement(x: np.ndarray) -> np.ndarray:
        return np.zeros_like(len(x))


class TemperatureDynamicProblem(DynamicProblem, TemperatureTimeDependentProblem, ABC):
    pass


class PiezoelectricQuasistaticProblem(
    QuasistaticDisplacementProblem, PiezoelectricTimeDependentProblem, ABC
):
    pass


class PiezoelectricDynamicProblem(
    DynamicDisplacementProblem, PiezoelectricTimeDependentProblem, ABC
):
    pass
