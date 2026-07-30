# CONMECH @ Jagiellonian University in Kraków
#
# Copyright (C) 2026  Piotr Bartman-Szwarc <piotr.bartman@uj.edu.pl>
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

import subprocess
import sys
import textwrap
from pathlib import Path

import numpy as np
import pytest
import sympy

from examples.common import error_norms as err
from examples.common import table_exporter

RNG = np.random.default_rng(20260130)


def structured_mesh(ih: int, scale=(2.0, 1.0)):
    """
    Structured triangulation with `ih` cells per unit length and uniform
    diagonals, so that doubling `ih` refines it exactly.
    """
    scale_x, scale_y = scale
    n_x, n_y = int(round(scale_x * ih)), int(round(scale_y * ih))
    grid_x, grid_y = np.meshgrid(
        np.linspace(0.0, scale_x, n_x + 1), np.linspace(0.0, scale_y, n_y + 1)
    )
    nodes = np.stack((grid_x.ravel(), grid_y.ravel()), axis=1)

    i_cell, j_cell = np.meshgrid(np.arange(n_x), np.arange(n_y), indexing="ij")
    i_cell, j_cell = i_cell.ravel(), j_cell.ravel()

    def idx(i, j):
        return i + (n_x + 1) * j

    lower = np.stack(
        (idx(i_cell, j_cell), idx(i_cell + 1, j_cell), idx(i_cell + 1, j_cell + 1)), axis=1
    )
    upper = np.stack(
        (idx(i_cell, j_cell), idx(i_cell + 1, j_cell + 1), idx(i_cell, j_cell + 1)), axis=1
    )
    return nodes, np.concatenate((lower, upper), axis=0)


def random_triangle():
    return np.array([[0.3, -0.2], [1.7, 0.1], [0.6, 1.4]]) + 0.1 * RNG.normal(size=(3, 2))


class FakeState:
    """Duck-typed stand-in for a state: `.body.mesh.nodes/.elements`, `.temperature`."""

    class _Mesh:
        def __init__(self, nodes, elements):
            self.nodes = nodes
            self.elements = elements

    class _Body:
        def __init__(self, nodes, elements):
            self.mesh = FakeState._Mesh(nodes, elements)

    def __init__(self, nodes, elements, temperature):
        self.body = FakeState._Body(nodes, elements)
        self.temperature = np.asarray(temperature, dtype=float).ravel()


def sympy_integral_over_mesh(nodes, elements, expression, symbols):
    """Exact integral of a polynomial over a triangulation, element by element."""
    s, t = sympy.symbols("s t", nonnegative=True)
    x_sym, y_sym = symbols
    total = sympy.Integer(0)
    for element in elements:
        p_0, p_1, p_2 = (sympy.Matrix(nodes[k]) for k in element)
        jacobian = sympy.Matrix.hstack(p_1 - p_0, p_2 - p_0).det()
        mapped = p_0 + (p_1 - p_0) * s + (p_2 - p_0) * t
        integrand = expression.subs({x_sym: mapped[0], y_sym: mapped[1]}, simultaneous=True)
        total += abs(jacobian) * sympy.integrate(
            sympy.integrate(integrand, (t, 0, 1 - s)), (s, 0, 1)
        )
    return float(sympy.nsimplify(total))


def _rough_field(points):
    """A field in no P1 space involved, so nothing comes out exact by accident."""
    return np.sin(1.3 * points[:, 0]) + 0.7 * points[:, 1] ** 3


def test_quadrature_weights_sum_to_one():
    assert abs(err._TRI_RULE_DEG5[:, 3].sum() - 1.0) < 1e-12
    assert np.allclose(err._TRI_RULE_DEG5[:, :3].sum(axis=1), 1.0, atol=1e-12)


def test_integral_of_one_is_the_area():
    for _ in range(5):
        nodes, elements = random_triangle(), np.array([[0, 1, 2]])
        area = err.element_areas(nodes, elements)[0]
        assert abs(err.integrate(nodes, elements, lambda pts: np.ones(len(pts))) - area) < 1e-12


def test_quadrature_is_exact_up_to_degree_five():
    nodes, elements = random_triangle(), np.array([[0, 1, 2]])
    x_sym, y_sym = sympy.symbols("x y")
    for i in range(6):
        for j in range(6 - i):
            expected = sympy_integral_over_mesh(
                nodes, elements, x_sym**i * y_sym**j, (x_sym, y_sym)
            )
            got = err.integrate(
                nodes, elements, lambda pts, i=i, j=j: pts[:, 0] ** i * pts[:, 1] ** j
            )
            assert abs(got - expected) < 1e-12 * max(1.0, abs(expected)), (i, j, got, expected)


def test_p1_gradient_of_a_linear_function():
    nodes, elements = structured_mesh(4)
    gradients = err.p1_gradients(nodes, elements, 3.0 * nodes[:, 0] - 2.0 * nodes[:, 1] + 7.0)
    assert np.allclose(gradients[:, 0], 3.0, atol=1e-12)
    assert np.allclose(gradients[:, 1], -2.0, atol=1e-12)


def test_no_error_for_a_linear_function():
    nodes, elements = structured_mesh(4)

    def exact(points):
        return 3.0 * points[:, 0] - 2.0 * points[:, 1] + 7.0

    def grad_exact(points):
        out = np.empty((len(points), 2))
        out[:, 0], out[:, 1] = 3.0, -2.0
        return out

    for key, value in err.errors_vs_exact(
        nodes, elements, exact(nodes), exact, grad_exact
    ).items():
        assert value < 1e-13, (key, value)


def test_interpolation_error_matches_an_exact_integral():
    """`u = 2 y^2 + 3 y` against a sympy integral, element by element."""
    nodes, elements = structured_mesh(2)

    def exact(points):
        return 2.0 * points[:, 1] ** 2 + 3.0 * points[:, 1]

    def grad_exact(points):
        out = np.zeros((len(points), 2))
        out[:, 1] = 4.0 * points[:, 1] + 3.0
        return out

    measured = err.errors_vs_exact(nodes, elements, exact(nodes), exact, grad_exact)

    x_sym, y_sym = sympy.symbols("x y")
    u_sym = 2 * y_sym**2 + 3 * y_sym
    gradients = err.p1_gradients(nodes, elements, exact(nodes))
    l2_sq = 0.0
    h1_sq = 0.0
    for index, element in enumerate(elements):
        matrix = sympy.Matrix([[1, nodes[k][0], nodes[k][1]] for k in element])
        values = sympy.Matrix(
            [u_sym.subs(y_sym, sympy.Rational(str(nodes[k][1]))) for k in element]
        )
        coefficients = matrix.solve(values)
        interpolant = coefficients[0] + coefficients[1] * x_sym + coefficients[2] * y_sym
        l2_sq += sympy_integral_over_mesh(
            nodes,
            np.array([element]),
            sympy.expand((u_sym - interpolant) ** 2),
            (x_sym, y_sym),
        )
        h1_sq += sympy_integral_over_mesh(
            nodes,
            np.array([element]),
            sympy.expand(
                (sympy.diff(u_sym, x_sym) - gradients[index, 0]) ** 2
                + (sympy.diff(u_sym, y_sym) - gradients[index, 1]) ** 2
            ),
            (x_sym, y_sym),
        )

    assert abs(measured["L2"] - np.sqrt(l2_sq)) < 1e-10
    assert abs(measured["H1_semi"] - np.sqrt(h1_sq)) < 1e-10
    assert abs(measured["V"] - np.sqrt(l2_sq + h1_sq)) < 1e-10


def test_error_between_a_state_and_itself_is_zero():
    nodes, elements = structured_mesh(4)
    state = FakeState(nodes, elements, RNG.normal(size=len(nodes)))
    measured = err.errors_between_states(state, state)
    assert measured["L2"] == 0.0
    assert measured["H1_semi"] == 0.0
    assert measured["V"] == 0.0


def test_nested_meshes_interpolate_exactly():
    """
    A coarse P1 function belongs to the fine space when the meshes are nested, so
    carrying it over changes neither its `L^2` nor its `H^1` norm.
    """
    coarse_nodes, coarse_elements = structured_mesh(4)
    fine_nodes, fine_elements = structured_mesh(8)
    coarse_values = _rough_field(coarse_nodes)

    consistency = err.interpolation_consistency_error(
        coarse_nodes, coarse_elements, coarse_values, fine_nodes, fine_elements
    )
    assert consistency["L2"] < 1e-12, consistency
    assert consistency["H1_semi"] < 1e-12, consistency

    projected = err.interpolate_p1(coarse_nodes, coarse_elements, coarse_values, fine_nodes)
    round_trip = err.errors_between_states(
        FakeState(coarse_nodes, coarse_elements, coarse_values),
        FakeState(fine_nodes, fine_elements, projected),
    )
    assert round_trip["V"] < 1e-12, round_trip


def test_non_nested_meshes_do_not():
    coarse_nodes, coarse_elements = structured_mesh(48)
    fine_nodes, fine_elements = structured_mesh(72)
    consistency = err.interpolation_consistency_error(
        coarse_nodes, coarse_elements, _rough_field(coarse_nodes), fine_nodes, fine_elements
    )
    assert consistency["L2"] > 1e-8, consistency
    assert consistency["H1_semi"] > 1e-8, consistency


def test_interpolation_outside_the_source_mesh_raises():
    nodes, elements = structured_mesh(4)
    outside = np.array([[5.0, 5.0], [6.0, 6.0]])
    with pytest.raises(ValueError, match="same domain"):
        err.interpolate_p1(nodes, elements, _rough_field(nodes), outside)


def test_rate_estimators():
    hs = [1 / 4, 1 / 8, 1 / 16, 1 / 32]
    for rate in err.rates_h([3.7 * h**2 for h in hs], hs)[1:]:
        assert abs(rate - 2.0) < 1e-9

    alphas = [1, 10, 100, 1000]
    for rate in err.rates_alpha([2.5 / a for a in alphas], alphas)[1:]:
        assert abs(rate - 1.0) < 1e-9

    assert not np.isfinite(err.rate_h(1.0, 0.0, 0.5, 0.25))
    assert not np.isfinite(err.rate_alpha(1.0, 0.5, 10.0, np.inf))


def test_formatters_reject_non_finite():
    assert table_exporter.format_error(None) == "--"
    assert table_exporter.format_error(float("nan")) == "--"
    assert table_exporter.format_rate(float("inf")) == "--"
    assert table_exporter.format_error(1.0) == "1.000e+00"
