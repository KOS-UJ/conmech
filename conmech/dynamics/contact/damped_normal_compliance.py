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
from conmech.dynamics.contact.contact_law import PotentialOfContactLaw
from conmech.dynamics.contact.interior_contact_law import InteriorContactLaw


def make_damped_norm_compl(obstacle_level: float, kappa: float, beta: float, interior=False):
    superclass = InteriorContactLaw if interior else PotentialOfContactLaw

    class DampedNormalCompliance(superclass):
        @property
        def kappa(self):
            return kappa

        @property
        def beta(self):
            return beta

        @property
        def obstacle_level(self):
            return obstacle_level

        @staticmethod
        def normal_bound(var_nu: float, static_displacement_nu: float, dt: float) -> float:
            """
            Since multiply by var_nu
            """
            return 0.5 * var_nu

        @staticmethod
        def potential_normal_direction(
            var_nu: float, static_displacement_nu: float, dt: float
        ) -> float:
            displacement = static_displacement_nu + var_nu * dt
            if displacement < obstacle_level:
                return 0
            return kappa * (displacement - obstacle_level) + beta * var_nu

        @staticmethod
        def subderivative_normal_direction(
            var_nu: float, static_displacement_nu: float, dt: float
        ) -> float:
            displacement = static_displacement_nu + var_nu * dt
            if displacement < obstacle_level:
                return 0
            return kappa * dt + beta

    return DampedNormalCompliance
