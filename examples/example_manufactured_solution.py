# CONMECH @ Jagiellonian University in Kraków
#
# Copyright (C) 2025-2026  Piotr Bartman <piotr.bartman@uj.edu.pl>
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
from dataclasses import dataclass
from typing import Type

import numpy as np

from conmech.dynamics.contact.contact_law import ContactLaw, PotentialOfContactLaw, DirectContactLaw
from conmech.helpers.config import Config
from conmech.mesh.boundaries_description import BoundariesDescription
from conmech.plotting.drawer import Drawer
from conmech.scenarios.problems import StaticDisplacementProblem
from conmech.simulations.problem_solver import StaticSolver
from conmech.properties.mesh_description import RectangleMeshDescription

# width and height of the rectangular domain
H = 1.0
W = 2.0
# constant in the manufactured solution
ALPHA = 0.05

# material properties
E = 12000.0
NU = 0.42

# Lamé parameters
LAMBDA = E * NU / ((1 + NU) * (1 - 2 * NU))
MU = E / (2 * (1 + NU))

# Contact stiffness matched to the material properties.
# Thanks to this, the normal contact reaction exactly balances.
C_N_MATCHED = LAMBDA + 2 * MU


def solution(x):
    """
    Exact solution of MMS:
    ux = alpha * (2xy + xy^2)
    uy = -alpha * x^2 * (1 + y)
    """
    res = np.zeros_like(x)
    # x[:,0] -> X, x[:,1] -> Y
    X_c = x[:, 0]
    Y_c = x[:, 1]

    res[:, 0] = ALPHA * (2 * X_c * Y_c + X_c * Y_c ** 2)
    res[:, 1] = -ALPHA * X_c ** 2 * (1 + Y_c)
    return res


def make_linear_contact_law(stiffness: float) -> Type[ContactLaw]:
    """linear spring: F = k*u"""

    class LinearPenaltyLaw(DirectContactLaw, PotentialOfContactLaw):
        @staticmethod
        def potential_normal_direction(
                var_nu: float, static_displacement_nu: float, dt: float
        ) -> float:
            if var_nu <= 0:
                return 0.0
            return 0.5 * stiffness * var_nu ** 2

        @staticmethod
        def potential_tangential_direction(var_tau, static_disp, dt) -> float:
            return 0.0

        @staticmethod
        def subderivative_normal_direction(
                var_nu: float, static_displacement_nu: float, dt: float
        ) -> float:
            if var_nu <= 0:
                return 0.0
            # F = stiffness * u_n
            return stiffness * var_nu

        @staticmethod
        def subderivative_tangential_direction(var_tau, static_disp, dt) -> float:
            return 0.0

    return LinearPenaltyLaw


@dataclass
class MMS_SelfBalancing(StaticDisplacementProblem):
    mu_coef: ... = MU
    la_coef: ... = LAMBDA
    contact_law: ... = make_linear_contact_law(stiffness=C_N_MATCHED)

    @staticmethod
    def inner_forces(x, v=None, t=None):
        """
        Dla u_x = A(2xy+xy^2), u_y = -Ax^2(1+y)
        f_x = 2 * lambda * A * x
        f_y = -2 * lambda * A * (1 + y)
        """
        X_c, Y_c = x[0], x[1]

        fx = 2 * LAMBDA * ALPHA * X_c
        fy = -2 * LAMBDA * ALPHA * (1 + Y_c)

        return np.array([fx, fy])

    @staticmethod
    def outer_forces(x, v=None, t=None):
        X_c, Y_c = x[0], x[1]

        result = np.array([0.0, 0.0])

        # 1. Right edge (x=W), n=[1, 0]
        # t_x = sigma_xx, t_y = sigma_xy = 0
        if X_c > W - 0.001:
            eps_xx = ALPHA * (2 * Y_c + Y_c ** 2)
            eps_yy = -ALPHA * X_c ** 2

            sig_xx = (LAMBDA + 2 * MU) * eps_xx + LAMBDA * eps_yy
            result += np.array([sig_xx, 0.0])

        # 2. Upper edge (y=H), n=[0, 1]
        # t_x = sigma_xy = 0, t_y = sigma_yy
        if Y_c > H - 0.001:
            eps_xx = ALPHA * (2 * Y_c + Y_c ** 2)
            eps_yy = -ALPHA * X_c ** 2

            sig_yy = LAMBDA * eps_xx + (LAMBDA + 2 * MU) * eps_yy
            result += np.array([0.0, sig_yy])

        if X_c > W - 0.001 and Y_c > H - 0.001:
            result /= 2.0

        return result

    boundaries: ... = BoundariesDescription(
        contact=lambda x: x[1] == 0, dirichlet=lambda x: x[0] == 0
    )


def calculate_l2_errors(elements, nodes, exact_vals, approx_vals):
    n_nodes = nodes.shape[0]

    diff = exact_vals - approx_vals  # [N_nodes, 2]

    # coords: [N_elem, 3, 2]
    tri_coords = nodes[elements]

    x1, y1 = tri_coords[:, 0, 0], tri_coords[:, 0, 1]
    x2, y2 = tri_coords[:, 1, 0], tri_coords[:, 1, 1]
    x3, y3 = tri_coords[:, 2, 0], tri_coords[:, 2, 1]

    # Area of each triangle element
    # 0.5 * |(x2-x1)(y3-y1) - (x3-x1)(y2-y1)|
    areas = 0.5 * np.abs((x2 - x1) * (y3 - y1) - (x3 - x1) * (y2 - y1))

    # Error vectors at the nodes of each element
    # diff_el: [N_elem, 3, 2]
    diff_el = diff[elements]
    d1 = diff_el[:, 0, :]
    d2 = diff_el[:, 1, :]
    d3 = diff_el[:, 2, :]

    # Consistent Mass Matrix
    # (Area/6) * ( sum(|di|^2) + sum(di . dj) )
    dot11 = np.sum(d1 * d1, axis=1)
    dot22 = np.sum(d2 * d2, axis=1)
    dot33 = np.sum(d3 * d3, axis=1)
    dot12 = np.sum(d1 * d2, axis=1)
    dot23 = np.sum(d2 * d3, axis=1)
    dot31 = np.sum(d3 * d1, axis=1)

    # element_integrals[i] = integral over ||e||^2
    element_integrals = (areas / 6.0) * (
            dot11 + dot22 + dot33 +
            dot12 + dot23 + dot31
    )

    total_l2_norm = np.sqrt(np.sum(element_integrals))

    # Smoothing
    node_error_sq_sum = np.zeros(n_nodes)
    node_area_sum = np.zeros(n_nodes)

    flat_elems = elements.ravel()
    flat_integrals = np.repeat(element_integrals, 3)
    flat_areas = np.repeat(areas, 3)

    np.add.at(node_error_sq_sum, flat_elems, flat_integrals)
    np.add.at(node_area_sum, flat_elems, flat_areas / 3.0)
    node_area_sum[node_area_sum == 0] = 1.0
    node_l2_errors = np.sqrt(node_error_sq_sum / node_area_sum)

    return node_l2_errors, total_l2_norm


def main(config: Config):
    solving_method = "schur"

    mesh_descr = RectangleMeshDescription(
        initial_position=None, max_element_perimeter=0.05, scale=[W, H]
    )

    setup = MMS_SelfBalancing(mesh_descr=mesh_descr)
    runner = StaticSolver(setup, solving_method)

    state = runner.solve(verbose=True, initial_displacement=setup.initial_displacement,
                         method="qsm")

    drawer = Drawer(state=state, config=config)

    displacement = solution(state.body.mesh.nodes)
    displacement = np.hstack((displacement[:, 0].reshape(-1, 1), displacement[:, 1].reshape(-1, 1)))
    drawer.initial_nodes = state.body.mesh.nodes + displacement

    # errors = drawer.initial_nodes - state.displaced_nodes
    errors, total = calculate_l2_errors(state.body.mesh.elements, state.body.mesh.nodes,
                                        drawer.initial_nodes, state.displaced_nodes)
    print(f"Total L2 error: {total}")
    # errors = np.linalg.norm(errors, axis=1)

    drawer.field = errors
    drawer.node_size = 0.01
    drawer.original_mesh_color = None
    drawer.deformed_mesh_color = None

    drawer.draw(show=config.show, save=config.save, field_max=np.max(drawer.field),
                field_min=np.min(drawer.field))


if __name__ == "__main__":
    show = True
    main(Config(save="output" if not show else False, show=show, force=False,
                output_dir="mms").init())
