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
import numpy as np
import pytest

from conmech.simulations.problem_solver import PoissonSolver
from examples.BartmanSzwarc_Ochal_Tarzia_2026 import setup as bot


def solve(alpha, ih):
    problem = bot.StaticPoissonSetup(bot.mesh_description(ih))
    if alpha == np.inf:
        problem.boundaries = bot.limit_boundaries()
        problem.contact_law_2 = None
    else:
        problem.contact_law_2 = bot.make_slope_contact_law(slope=alpha)
    runner = PoissonSolver(problem, bot.solving_method(alpha))
    return runner.solve(verbose=False, method="Powell"), runner


def traces(state):
    nodes = np.asarray(state.body.mesh.nodes, dtype=float)
    values = np.asarray(state.temperature, dtype=float).ravel()
    return (
        values[np.isclose(nodes[:, 1], 0.0)],
        values[np.isclose(nodes[:, 1], 1.0)],
    )


def test_limit_problem_uses_a_position_dependent_dirichlet_value():
    state, _ = solve(np.inf, 4)
    gamma1, gamma3 = traces(state)
    assert gamma1.size and gamma3.size
    assert np.allclose(gamma1, bot.GAMMA1_VALUE, atol=1e-10)
    assert np.allclose(gamma3, bot.B_COEF, atol=1e-10)


def test_finite_alpha_constrains_gamma1_only():
    state, _ = solve(100.0, 4)
    gamma1, gamma3 = traces(state)
    assert np.allclose(gamma1, bot.GAMMA1_VALUE, atol=1e-10)
    assert not np.allclose(gamma3, bot.B_COEF, atol=1e-3)
    assert np.all(gamma3 < bot.B_COEF)


def test_limit_problem_is_one_sparse_solve():
    _, runner = solve(np.inf, 8)
    assert str(runner.second_step_solver) == "direct"
    assert runner.second_step_solver.equation is None


@pytest.mark.parametrize("slope", [1.0, 3.0, 1000.0])
def test_subderivative_is_the_derivative_of_the_potential(slope):
    """Central difference of `j` against `subderivative_normal_direction`."""
    law = bot.make_slope_contact_law(slope)
    b = bot.B_COEF
    step = 1e-6
    for r in np.linspace(b - 3.0, b + 3.0, 601):
        r = float(r)
        if abs(r - b) < 10 * step:  # skip the non-differentiable point
            continue
        finite_difference = (
            law.potential_normal_direction(r + step, 0.0, 0.0)
            - law.potential_normal_direction(r - step, 0.0, 0.0)
        ) / (2 * step)
        analytic = law.subderivative_normal_direction(r, 0.0, 0.0)
        assert abs(finite_difference - analytic) < 1e-6 * max(1.0, abs(analytic)), r
