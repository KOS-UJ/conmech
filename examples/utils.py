# CONMECH @ Jagiellonian University in Kraków
#
# Copyright (C) 2023  Piotr Bartman <piotr.bartman@uj.edu.pl>
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
from typing import List

import numba
import numpy as np
import scipy

from conmech.properties.body_properties import BodyProperties


def get_interpolated(u, vertices):
    x = vertices[:, 0].copy()
    y = vertices[:, 1].copy()
    u_x = u[:, 0].copy()
    u_y = u[:, 1].copy()
    u_fun_x = scipy.interpolate.LinearNDInterpolator(list(zip(x, y)), u_x)
    u_fun_y = scipy.interpolate.LinearNDInterpolator(list(zip(x, y)), u_y)
    return u_fun_x, u_fun_y


@numba.njit()
def calculate_dx_dy(x0, u0, x1, u1, x2, u2):
    a1 = x1[0] - x0[0]
    b1 = x1[1] - x0[1]
    c1 = u1 - u0
    a2 = x2[0] - x0[0]
    b2 = x2[1] - x0[1]
    c2 = u2 - u0
    a = b1 * c2 - b2 * c1
    b = a2 * c1 - a1 * c2
    c = a1 * b2 - b1 * a2
    dx = a / c
    dy = b / c
    return dx, dy


def gradient(elements, initial_nodes, f):
    result = np.zeros((len(f), 2))
    norm = np.zeros(len(f))
    for element in elements:
        x0 = initial_nodes[element[0]]
        x1 = initial_nodes[element[1]]
        x2 = initial_nodes[element[2]]
        f0 = f[element[0]]
        f1 = f[element[1]]
        f2 = f[element[2]]
        dx, dy = calculate_dx_dy(x0, f0, x1, f1, x2, f2)
        result[element[0], 0] += dx
        result[element[0], 1] += dy
        norm[element[0]] += 1
        result[element[1], 0] += dx
        result[element[1], 1] += dy
        norm[element[1]] += 1
        result[element[2], 0] += dx
        result[element[2], 1] += dy
        norm[element[2]] += 1

    result[:, 0] /= norm
    result[:, 1] /= norm

    return result


def viscoelastic_constitutive_law(displacement, velocity, setup, elements, nodes, **_kwargs):
    grad_x = gradient(elements, nodes, displacement[:, 0])
    grad_y = gradient(elements, nodes, displacement[:, 1])
    grad_u = np.concatenate((grad_x, grad_y), axis=1).reshape(-1, 2, 2)

    stress_u = np.zeros_like(grad_u)
    stress_u[:, 0, 0] = 2 * setup.mu_coef * grad_u[:, 0, 0] + setup.la_coef * (
        grad_u[:, 0, 0] + grad_u[:, 1, 1]
    )
    stress_u[:, 1, 1] = 2 * setup.mu_coef * grad_u[:, 1, 1] + setup.la_coef * (
        grad_u[:, 0, 0] + grad_u[:, 1, 1]
    )
    stress_u[:, 0, 1] = setup.mu_coef * (grad_u[:, 0, 1] + grad_u[:, 1, 0])
    stress_u[:, 1, 0] = stress_u[:, 0, 1]

    grad_x = gradient(elements, nodes, velocity[:, 0])
    grad_y = gradient(elements, nodes, velocity[:, 1])
    grad_v = np.concatenate((grad_x, grad_y), axis=1).reshape(-1, 2, 2)

    stress_v = np.zeros_like(grad_v)
    stress_v[:, 0, 0] = 2 * setup.th_coef * grad_v[:, 0, 0] + setup.ze_coef * (
        grad_v[:, 0, 0] + grad_v[:, 1, 1]
    )
    stress_v[:, 1, 1] = 2 * setup.th_coef * grad_v[:, 1, 1] + setup.ze_coef * (
        grad_v[:, 0, 0] + grad_v[:, 1, 1]
    )
    stress_v[:, 0, 1] = setup.th_coef * (grad_v[:, 0, 1] + grad_v[:, 1, 0])
    stress_v[:, 1, 0] = stress_v[:, 0, 1]
    return stress_u + stress_v


def elastic_relaxation_constitutive_law(
    displacement: np.ndarray, absement: np.ndarray, setup, elements, nodes, time, **_kwargs
):  # TODO!
    grad_x = gradient(elements, nodes, displacement[:, 0])
    grad_y = gradient(elements, nodes, displacement[:, 1])
    grad_u = np.concatenate((grad_x, grad_y), axis=1).reshape(-1, 2, 2)

    stress_u = np.zeros_like(grad_u)
    stress_u[:, 0, 0] = 2 * setup.mu_coef * grad_u[:, 0, 0] + setup.la_coef * (
        grad_u[:, 0, 0] + grad_u[:, 1, 1]
    )
    stress_u[:, 1, 1] = 2 * setup.mu_coef * grad_u[:, 1, 1] + setup.la_coef * (
        grad_u[:, 0, 0] + grad_u[:, 1, 1]
    )
    stress_u[:, 0, 1] = setup.mu_coef * (grad_u[:, 0, 1] + grad_u[:, 1, 0])
    stress_u[:, 1, 0] = stress_u[:, 0, 1]

    rlx_coef = setup.relaxation(time)[1, 0, 1]  # TODO

    stress_b = np.zeros_like(grad_u)
    grad_x = gradient(elements, nodes, absement[:, 0])
    grad_y = gradient(elements, nodes, absement[:, 1])
    grad_v = np.concatenate((grad_x, grad_y), axis=1).reshape(-1, 2, 2)

    stress_b[:, 0, 0] += 2 * rlx_coef * grad_v[:, 0, 0]
    stress_b[:, 1, 1] += 2 * rlx_coef * grad_v[:, 1, 1]
    stress_b[:, 0, 1] += rlx_coef * (grad_v[:, 0, 1] + grad_v[:, 1, 0])
    stress_b[:, 1, 0] += stress_b[:, 0, 1]
    return stress_u + stress_b
