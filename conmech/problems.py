"""
Contact Mechanics Problem setups
"""

from dataclasses import dataclass
from typing import Union, Tuple
import numpy as np


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
    grid_height: float

    elements_number: Union[
        Tuple[int, int], Tuple[int, int, int]
    ]  # number of triangles per aside

    mu_coef: float
    la_coef: float
    contact_law: ContactLaw
    dynamism: str = None  # TODO: remove

    @staticmethod
    def inner_forces(x: float, y: float) -> np.ndarray:
        raise NotImplementedError()

    @staticmethod
    def outer_forces(x: float, y: float) -> np.ndarray:
        raise NotImplementedError()

    @staticmethod
    def friction_bound(u_nu: float) -> float:
        raise NotImplementedError()

    @staticmethod
    def is_contact(x: np.ndarray) -> bool:
        raise NotImplementedError()

    @staticmethod
    def is_dirichlet(x: np.ndarray) -> bool:
        raise NotImplementedError()


class Static(Problem):
    @staticmethod
    def inner_forces(x: float, y: float) -> np.ndarray:
        raise NotImplementedError()

    @staticmethod
    def outer_forces(x: float, y: float) -> np.ndarray:
        raise NotImplementedError()

    @staticmethod
    def friction_bound(u_nu: float) -> float:
        raise NotImplementedError()

    @staticmethod
    def is_contact(x: np.ndarray) -> bool:
        raise NotImplementedError()

    @staticmethod
    def is_dirichlet(x: np.ndarray) -> bool:
        raise NotImplementedError()


class Quasistatic(Problem):
    th_coef: float
    ze_coef: float
    time_step: float

    @staticmethod
    def inner_forces(x: float, y: float) -> np.ndarray:
        raise NotImplementedError()

    @staticmethod
    def outer_forces(x: float, y: float) -> np.ndarray:
        raise NotImplementedError()

    @staticmethod
    def friction_bound(u_nu: float) -> float:
        raise NotImplementedError()

    @staticmethod
    def is_contact(x: np.ndarray) -> bool:
        raise NotImplementedError()

    @staticmethod
    def is_dirichlet(x: np.ndarray) -> bool:
        raise NotImplementedError()


class Dynamic(Problem):
    th_coef: float
    ze_coef: float
    time_step: float

    @staticmethod
    def inner_forces(x: float, y: float) -> np.ndarray:
        raise NotImplementedError()

    @staticmethod
    def outer_forces(x: float, y: float) -> np.ndarray:
        raise NotImplementedError()

    @staticmethod
    def friction_bound(u_nu: float) -> float:
        raise NotImplementedError()

    @staticmethod
    def is_contact(x: np.ndarray) -> bool:
        raise NotImplementedError()

    @staticmethod
    def is_dirichlet(x: np.ndarray) -> bool:
        raise NotImplementedError()
