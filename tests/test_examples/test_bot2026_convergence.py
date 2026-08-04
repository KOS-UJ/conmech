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
from examples.common import error_norms as err


def solve(spec, alpha, ih):
    problem, method = bot.build_setup(spec, alpha, bot.mesh_description(ih))
    return PoissonSolver(problem, method).solve(verbose=False, method="Powell")


def errors(state, spec, alpha):
    u_exact, grad_exact = spec.exact_for(alpha)
    return err.errors_vs_exact(
        state.body.mesh.nodes, state.body.mesh.elements, state.temperature, u_exact, grad_exact
    )


@pytest.mark.parametrize("spec_name", ["example1_1d", "example1_2d"])
def test_convergence_in_h_for_the_limit_problem(spec_name):
    spec = bot.EXAMPLES[spec_name]
    ihs = [4, 8, 16]
    hs = [bot.mesh_size(ih) for ih in ihs]
    measured = [errors(solve(spec, np.inf, ih), spec, np.inf) for ih in ihs]

    rates_l2 = err.rates_h([m["L2"] for m in measured], hs)[1:]
    rates_h1 = err.rates_h([m["H1_semi"] for m in measured], hs)[1:]
    assert all(1.5 <= rate <= 2.3 for rate in rates_l2), rates_l2
    assert all(0.8 <= rate <= 1.3 for rate in rates_h1), rates_h1


@pytest.mark.parametrize("alpha", [1.0, 10.0, 100.0])
def test_robin_condition_is_scaled_by_alpha(alpha):
    ih = 16
    spec = bot.EXAMPLE_1_1D
    measured = errors(solve(spec, alpha, ih), spec, alpha)["V"]
    yardstick = errors(solve(spec, np.inf, ih), spec, np.inf)["V"]
    assert measured < 1.05 * yardstick, (measured, yardstick)

    if alpha > 1.0:
        state = solve(spec, alpha, ih)
        wrong_scale = err.errors_vs_exact(
            state.body.mesh.nodes,
            state.body.mesh.elements,
            state.temperature,
            bot.u_exact_1d(1.0),
            bot.grad_u_exact_1d(1.0),
        )["V"]
        assert measured < 0.1 * wrong_scale, (measured, wrong_scale)


def test_convergence_in_alpha():
    ih = 16
    alphas = [1.0, 10.0, 100.0, 1000.0]
    spec = bot.EXAMPLE_1_1D
    reference = solve(spec, np.inf, ih)
    measured = [
        err.errors_between_states(solve(spec, alpha, ih), reference)["L2"] for alpha in alphas
    ]

    for alpha, value in zip(alphas, measured):
        expected = bot.analytic_gap_l2(alpha)
        assert abs(value - expected) < 0.05 * expected, (alpha, value, expected)

    # the O(1/alpha) rate is asymptotic, so only the tail is pinned to 1
    for rate in err.rates_alpha(measured, alphas)[2:]:
        assert 0.9 < rate < 1.1, (measured, rate)


def test_trace_approaches_b_like_one_over_alpha():
    ih = 8
    alphas = [10.0, 100.0, 1000.0]
    gaps = []
    for alpha in alphas:
        state = solve(bot.EXAMPLE_1_1D, alpha, ih)
        nodes = np.asarray(state.body.mesh.nodes, dtype=float)
        values = np.asarray(state.temperature, dtype=float).ravel()
        trace = values[np.isclose(nodes[:, 1], 1.0)]
        gaps.append(bot.B_COEF - trace.max())
        # and it matches the closed-form trace
        assert abs(trace.max() - bot.analytic_trace_1d(alpha)) < 0.05 * gaps[-1]

    assert all(gap > 0 for gap in gaps), gaps
    for rate in err.rates_alpha(gaps, alphas)[1:]:
        assert 0.9 < rate < 1.1, (gaps, rate)


def test_meshes_are_nested():
    built = {}
    for ih in (4, 8):
        problem, method = bot.build_setup(bot.EXAMPLE_1_1D, 100.0, bot.mesh_description(ih))
        mesh = PoissonSolver(problem, method).body.mesh
        built[ih] = (np.asarray(mesh.nodes, dtype=float), np.asarray(mesh.elements, dtype=int))

    coarse_nodes, coarse_elements = built[4]
    fine_nodes, fine_elements = built[8]
    values = np.sin(1.3 * coarse_nodes[:, 0]) + 0.7 * coarse_nodes[:, 1] ** 3
    consistency = err.interpolation_consistency_error(
        coarse_nodes, coarse_elements, values, fine_nodes, fine_elements
    )
    assert consistency["V"] < 1e-12, consistency
