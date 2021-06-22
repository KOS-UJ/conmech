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
    def regularized_subderivative_tangential_direction(u_tau: np.ndarray, v_tau: np.ndarray, rho=1e-7) -> float:
        """
        Coulomb regularization
        """
        raise NotImplementedError()


@dataclass()
class Problem:
    grid_height: int
    cells_number: Union[Tuple[int, int], Tuple[int, int, int]]  # number of triangles per aside
    inner_forces: Union[Tuple[float, float], Tuple[float, float, float]]
    outer_forces: Union[Tuple[float, float], Tuple[float, float, float]]
    mu_coef: float
    lambda_coef: float
    contact_law: ContactLaw
    dynamism: str = None  # TODO: remove

    @staticmethod
    def friction_bound(u_nu):
        raise NotImplementedError()


class Static(Problem):
    @staticmethod
    def friction_bound(u_nu):
        raise NotImplementedError()


class Quasistatic(Problem):
    th_coef: float
    ze_coef: float
    time_step: float

    @staticmethod
    def friction_bound(u_nu):
        raise NotImplementedError()


class Dynamic(Problem):
    th_coef: float
    ze_coef: float
    time_step: float

    @staticmethod
    def friction_bound(u_nu):
        raise NotImplementedError()
