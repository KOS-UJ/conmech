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
import matplotlib.tri as tri
import scipy.optimize as opt

from conmech.state.products.product import Product


class IntersectionContactLimitPoints(Product):

    def __init__(self, x, obstacle_level):
        super().__init__(f'limit points at {x:.2f}')
        self.x = x
        self.obstacle_level = obstacle_level

    def update(self, state):
        y_min = np.min(state.body.mesh.nodes[:, 1])
        y_max = np.max(state.body.mesh.nodes[:, 1])
        step = len(state.body.mesh.nodes) ** -1 / 2
        u = interpolate(state, 'displacement')

        def u_intsec(y):
            return u(self.x, y) - self.obstacle_level

        self.data[state.time] = estimate_zeros(u_intsec, y_min, y_max, step)


def interpolate(state, field):
    assert state.body.mesh.dimension == 2  # membrane have to be 2D
    X = state.body.mesh.nodes[:, 0]
    Y = state.body.mesh.nodes[:, 1]

    soltri = tri.Triangulation(X, Y, triangles=state.body.mesh.elements)
    interpol = tri.LinearTriInterpolator
    return interpol(soltri, getattr(state, field)[:, 0])


def estimate_zeros(f, xmin, xmax, step) -> tuple:
    zeros = []
    x = xmin
    sign = np.sign(f(x))
    while x <= xmax:
        f_x = f(x)
        if f_x == 0:
            zeros.append(x)
        elif f_x < 0 < sign or f_x > 0 > sign:
            zeros.append(opt.brentq(f, x-step, x))
        sign = np.sign(f_x)
        x += step
    return tuple(zeros)
