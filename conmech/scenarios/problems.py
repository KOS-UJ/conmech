"""
Contact Mechanics Problem setups
"""
from abc import ABC
from dataclasses import dataclass
from typing import Tuple, Union, Optional

import numpy as np

from conmech.mesh.boundaries_description import BoundariesDescription


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


@dataclass()
class Problem:
    dimension = 2  # TODO #74 : Not used?
    mesh_type: str
    grid_height: float
    boundaries: BoundariesDescription

    elements_number: Union[Tuple[int, int], Tuple[int, int, int]]  # number of triangles per aside

    mu_coef: float
    la_coef: float
    contact_law: ContactLaw
    dynamism: str = None  # TODO: #65 remove

    @staticmethod
    def initial_displacement(x: np.ndarray) -> np.ndarray:
        return np.zeros_like(x)

    @staticmethod
    def inner_forces(x: np.ndarray, t: Optional[float] = None) -> np.ndarray:
        raise NotImplementedError()

    @staticmethod
    def outer_forces(x: np.ndarray, t: Optional[float] = None) -> np.ndarray:
        raise NotImplementedError()

    @staticmethod
    def friction_bound(u_nu: float) -> float:
        raise NotImplementedError()


class Static(Problem):
    @staticmethod
    def inner_forces(x: np.ndarray, t: Optional[float] = None) -> np.ndarray:
        raise NotImplementedError()

    @staticmethod
    def outer_forces(x: np.ndarray, t: Optional[float] = None) -> np.ndarray:
        raise NotImplementedError()

    @staticmethod
    def friction_bound(u_nu: float) -> float:
        raise NotImplementedError()


class TimeDependent(Problem):
    th_coef: float
    ze_coef: float
    time_step: float

    @staticmethod
    def initial_velocity(x: np.ndarray) -> np.ndarray:
        return np.zeros_like(x)

    @staticmethod
    def inner_forces(x: np.ndarray, t: Optional[float] = None) -> np.ndarray:
        raise NotImplementedError()

    @staticmethod
    def outer_forces(x: np.ndarray, t: Optional[float] = None) -> np.ndarray:
        raise NotImplementedError()

    @staticmethod
    def friction_bound(u_nu: float) -> float:
        raise NotImplementedError()


class Quasistatic(TimeDependent, ABC):
    pass


class Dynamic(TimeDependent, ABC):
    pass


class TemperatureTimeDependent(TimeDependent, ABC):
    thermal_expansion: np.ndarray
    thermal_conductivity: np.ndarray

    @staticmethod
    def initial_temperature(x: np.ndarray) -> np.ndarray:
        return np.zeros_like(len(x))


class PiezoelectricTimeDependent(TimeDependent, ABC):
    piezoelectricity: np.ndarray
    permittivity: np.ndarray

    @staticmethod
    def initial_electric_potential(x: np.ndarray) -> np.ndarray:
        return np.zeros_like(len(x))


class RelaxationQuasistaticProblem(Quasistatic, ABC):
    relaxation: np.ndarray

    @staticmethod
    def initial_absement(x: np.ndarray) -> np.ndarray:
        return np.zeros_like(len(x))


class TemperatureDynamic(Dynamic, TemperatureTimeDependent, ABC):
    pass


class PiezoelectricQuasistatic(Quasistatic, PiezoelectricTimeDependent, ABC):
    pass


class PiezoelectricDynamic(Dynamic, PiezoelectricTimeDependent, ABC):
    pass
