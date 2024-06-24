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

import numpy as np

from conmech.state.products.product import Product


class Penetration(Product):
    """
    This class assume foundation equals x=0.
    """

    def __init__(self):
        super().__init__('penetration')

    def update(self, state):
        if len(state.displaced_nodes[state.body.mesh.contact_indices, 1]) != 0:
            all_p = state.displaced_nodes[state.body.mesh.contact_indices, 1]
            p = np.min(all_p)
        else:
            p = 0.
        self.data[state.time] = p
