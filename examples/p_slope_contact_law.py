"""
Example contact law
"""
from typing import Type

import numpy as np

from conmech.problems import ContactLaw


def make_slope_contact_law(slope: float) -> Type[ContactLaw]:
    class PSlopeContactLaw(ContactLaw):
        @staticmethod
        def potential_normal_direction(u_nu: float) -> float:
            if u_nu <= 0:
                return 0.
            return (0.5 * slope * u_nu) * u_nu

        @staticmethod
        def subderivative_normal_direction(u_nu: float, v_nu: float) -> float:
            if u_nu <= 0:
                return 0 * v_nu
            return (slope * u_nu) * v_nu

        @staticmethod
        def regularized_subderivative_tangential_direction(u_tau: np.ndarray, v_tau: np.ndarray,
                                                           rho=1e-7) -> float:
            """
            Coulomb regularization
            """
            regularization = 1 / np.sqrt(u_tau[0] * u_tau[0] + u_tau[1] * u_tau[1] + rho ** 2)
            result = regularization * (u_tau[0] * v_tau[0] + u_tau[1] * v_tau[1])
            return result

    return PSlopeContactLaw
