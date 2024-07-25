# CONMECH @ Jagiellonian University in Kraków
#
# Copyright (C) 2024  Piotr Bartman-Szwarc <piotr.bartman@uj.edu.pl>
#
# This program is free software; you can redistribute it and/or
# modify it under the terms of the GNU General Public License
# as published by the Free Software Foundation; either version 3
# of the License, or (at your option) any later version.
#
# This program is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
# GNU General Public License for more details.
#
# You should have received a copy of the GNU General Public License
# along with this program; if not, write to the Free Software
# Foundation, Inc., 51 Franklin Street, Fifth Floor, Boston, MA  02110-1301,
# USA.
from abc import ABC, abstractmethod


class ContactLaw:
    # pylint: disable=unused-argument)
    """
    "Abstract" class for all contact conditions.
    """

    @staticmethod
    def normal_bound(var_nu: float, static_displacement_nu: float, dt: float) -> float:
        """
        :param var_nu: variable normal vector length
        :param static_displacement_nu: normal vector length of displacement from
         the previous step. For time independent problems this likely be 0.
        :param dt: time step
        :returns foundation response
        """
        return 1.0

    @staticmethod
    def tangential_bound(var_nu: float, static_displacement_nu: float, dt: float) -> float:
        """
        Friction bound

        :param var_nu: variable normal vector length
        :param static_displacement_nu: normal vector length of displacement from
         the previous step. For time independent problems this likely be 0.
        :param dt: time step
        :returns foundation response
        """
        return 1.0


class DirectContactLaw(ContactLaw, ABC):
    # pylint: disable=unused-argument)
    """
    Abstract class for contact conditions given in a direct form.
    Since usually contact law is a multifunction it can be treated as
    a subderivative of *some* function - potential. Hence, from point of view
    of conmech package we call subderivative form.
    """

    @staticmethod
    @abstractmethod
    def subderivative_normal_direction(
        var_nu: float, static_displacement_nu: float, dt: float
    ) -> float:
        """
        :param var_nu: variable normal vector length
        :param static_displacement_nu: normal vector length of displacement from
         the previous step. For time independent problems this likely be 0.
        :param dt: time step
        :returns foundation response
        """
        raise NotImplementedError()

    @staticmethod
    def subderivative_tangential_direction(
        var_tau: float, static_displacement_tau: float, dt: float
    ) -> float:
        """
        :param var_tau: variable normal vector length
        :param static_displacement_tau: normal vector length of displacement from
         the previous step. For time independent problems this likely be 0.
        :param dt: time step
        :returns potential of foundation friction response
        """
        return 0.0


class PotentialOfContactLaw(ContactLaw, ABC):
    # pylint: disable=unused-argument)
    """
    Abstract class for contact conditions given in a potential form.
    Since usually contact law is a multifunction it can be treated as
    a subderivative of *some* function - potential. To solve contact problem
    numerically with optimization approach we need only the potential of
    contact condition. Hence, from point of view of conmech package,
    we call potential form.
    """

    @staticmethod
    @abstractmethod
    def potential_normal_direction(
        var_nu: float, static_displacement_nu: float, dt: float
    ) -> float:
        """
        :param var_nu: variable normal vector length
        :param static_displacement_nu: normal vector length of displacement from
         the previous step. For time independent problems this likely be 0.
        :param dt: time step
        :returns potential of foundation response
        """
        raise NotImplementedError()

    @staticmethod
    def potential_tangential_direction(
        var_tau: float, static_displacement_tau: float, dt: float
    ) -> float:
        """
        :param var_tau: variable normal vector length
        :param static_displacement_tau: normal vector length of displacement from
         the previous step. For time independent problems this likely be 0.
        :param dt: time step
        :returns potential of foundation friction response
        """
        return 0
