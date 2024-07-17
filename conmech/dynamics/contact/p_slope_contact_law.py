"""
Example contact law
"""

from typing import Type

import numpy as np
from conmech.dynamics.contact.contact_law import ContactLaw, DirectContactLaw, \
    PotentialOfContactLaw


def make_slope_contact_law(slope: float) -> Type[ContactLaw]:
    class PSlopeContactLaw(DirectContactLaw, PotentialOfContactLaw):
        @staticmethod
        def potential_normal_direction(
                var_nu: float,
                static_displacement_nu: float,
                dt: float
        ) -> float:
            if var_nu <= 0:
                return 0.0
            return 0.5 * slope * var_nu ** 2

        @staticmethod
        def subderivative_normal_direction(
                var_nu: float,
                static_displacement_nu: float,
                dt: float
        ) -> float:
            if var_nu <= 0:
                return 0.0
            return slope * var_nu * static_displacement_nu

    return PSlopeContactLaw


def make_const_contact_law(slope: float) -> Type[ContactLaw]:
    class PSlopeContactLaw(DirectContactLaw, PotentialOfContactLaw):
        @staticmethod
        def potential_normal_direction(
                var_nu: float,
                static_displacement_nu: float,
                dt: float
        ) -> float:
            if var_nu <= 0:
                return 0 * static_displacement_nu
            return (slope * var_nu) * static_displacement_nu

        @staticmethod
        def subderivative_normal_direction(
                var_nu: float,
                static_displacement_nu: float,
                dt: float
        ) -> float:
            if var_nu <= 0:
                return 0
            return slope

    return PSlopeContactLaw
