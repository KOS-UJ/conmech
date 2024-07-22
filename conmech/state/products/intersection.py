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
from matplotlib import tri

from conmech.state.products.product import Product


class Intersection(Product):
    def __init__(self, x):
        super().__init__(f"intersection at {x:.2f}")
        self.x = x

    def update(self, state):
        y_min = np.min(state.body.mesh.nodes[:, 1])
        y_max = np.max(state.body.mesh.nodes[:, 1])
        step = len(state.body.mesh.nodes) ** -1 / 2
        space = np.linspace(y_min, y_max, int((y_max - y_min) // step))

        X = state.body.mesh.nodes[:, 0]
        Y = state.body.mesh.nodes[:, 1]
        soltri = tri.Triangulation(X, Y, triangles=state.body.mesh.elements)
        u = tri.LinearTriInterpolator(soltri, state.displacement[:, 0])

        self.data[state.time] = (space, u(np.ones_like(space) * self.x, space))
