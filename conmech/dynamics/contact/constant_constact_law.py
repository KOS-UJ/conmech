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
from typing import Type

from conmech.dynamics.contact.contact_law import ContactLaw, DirectContactLaw, \
    PotentialOfContactLaw


def make_const_contact_law(resistance: float) -> Type[ContactLaw]:
    class PSlopeContactLaw(DirectContactLaw, PotentialOfContactLaw):
        @staticmethod
        def potential_normal_direction(
                var_nu: float,
                static_displacement_nu: float,
                dt: float
        ) -> float:
            if var_nu <= 0:
                return 0 * static_displacement_nu
            return (resistance * var_nu) * static_displacement_nu

        @staticmethod
        def subderivative_normal_direction(
                var_nu: float,
                static_displacement_nu: float,
                dt: float
        ) -> float:
            if var_nu <= 0:
                return 0
            return resistance

    return PSlopeContactLaw
