# CONMECH @ Jagiellonian University in Kraków
#
# Copyright (C) 2025-2026  Piotr Bartman-Szwarc <piotr.bartman@uj.edu.pl>
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
"""
Simulations for the steady-state heat conduction example.
"""

from pathlib import Path
from typing import Optional

import numpy as np

from conmech.helpers.config import Config
from conmech.simulations.problem_solver import PoissonSolver
from conmech.state.state import TemperatureState

from examples.BartmanSzwarc_Ochal_Tarzia_2026 import setup as setup_module
from examples.BartmanSzwarc_Ochal_Tarzia_2026.setup import (
    ALPHAS_FINITE,
    ALPHAS_MATRIX,
    ALPHAS_TRACE,
    EXAMPLE_1_1D,
    EXAMPLE_1_2D,
    EXAMPLE_2,
    IH_REF,
    IHS,
    IHS_EXACT,
    TEMPERATURE_GRID,
    ExampleSpec,
    alpha_tag,
    build_setup,
    mesh_description,
)
from examples.common.runner import run_example
from examples.common.simulation_cache import SimulationCache


def cache(config, spec: ExampleSpec) -> SimulationCache:
    return SimulationCache(setup_module, Path(config.outputs_path) / spec.name)


def state_path(config, spec: ExampleSpec, alpha: float, ih: int) -> Path:
    return cache(config, spec).path(alpha=alpha_tag(alpha), ih=ih)


def simulate(config, spec: ExampleSpec, alpha: float, ih: int) -> TemperatureState:
    print(f"Simulate {spec.name}: alpha={alpha}, ih={ih}")
    setup, solving_method = build_setup(spec, alpha, mesh_description(ih))
    runner = PoissonSolver(setup, solving_method)
    state = runner.solve(verbose=False, method="Powell")

    # these carry unpicklable closures
    state.body.dynamics.force.outer.source = None
    state.body.dynamics.force.inner.source = None
    state.body.dynamics.temperature.outer.source = None
    state.body.dynamics.temperature.inner.source = None
    state.body.properties.relaxation = None
    state.setup = None
    state.constitutive_law = None

    if config.outputs_path:
        cache(config, spec).save(state, alpha=alpha_tag(alpha), ih=ih)
    return state


def load_or_simulate(
    config, spec: ExampleSpec, alpha: float, ih: int, only_ensure: bool = False
) -> Optional[TemperatureState]:
    stored = cache(config, spec)
    if config.force or not stored.is_current(alpha=alpha_tag(alpha), ih=ih):
        state = simulate(config, spec, alpha, ih)
        return None if only_ensure else state
    if only_ensure:
        return None
    return stored.load(alpha=alpha_tag(alpha), ih=ih)


def main(config: Config):
    """
    Entrypoint to example.

    To see result of simulation you need to call from python `main(Config().init())`.
    """
    Path(config.outputs_path).mkdir(parents=True, exist_ok=True)

    # imported here, not at module level: postprocess reads load_or_simulate
    # from this module, TODO: move common parts in one place
    from examples.BartmanSzwarc_Ochal_Tarzia_2026 import postprocess

    if config.test:
        ihs_exact, ihs = [4, 8], [4, 8]
        alphas_c, alphas_matrix, alphas_trace = [1, 10], [1, 10], [1, 10]
        ih_gap, alpha_b, ih_ref = 8, 10, 16
    else:
        ihs_exact, ihs = IHS_EXACT, IHS
        alphas_c, alphas_matrix, alphas_trace = ALPHAS_FINITE, ALPHAS_MATRIX, ALPHAS_TRACE
        ih_gap, alpha_b, ih_ref = 64, 10_000, IH_REF

    # A: the limit problem against the 2D closed-form solution
    postprocess.table_vs_exact(config, EXAMPLE_1_2D, np.inf, ihs_exact, table_id="A")
    # B: a fixed finite alpha against the exact u_alpha
    postprocess.table_vs_exact(config, EXAMPLE_1_1D, alpha_b, ihs_exact, table_id="B")
    # C: the order in alpha at fixed h
    postprocess.table_alpha_gap(config, EXAMPLE_1_1D, ih_gap, alphas_c)
    # D: the double limit
    postprocess.table_double_limit(config, EXAMPLE_1_1D, ihs, alphas_matrix)
    # B2: the 2D example at a finite alpha. It has a closed form for the limit
    # problem only, so its rates there can only be seen against a reference.
    postprocess.table_vs_reference(config, EXAMPLE_1_2D, alpha_b, ihs, ih_ref, table_id="B2")
    # E, F
    postprocess.figure_alpha_paths(config, EXAMPLE_1_1D, ihs)
    postprocess.figure_gamma3_trace(config, EXAMPLE_1_1D, ih_gap, alphas_trace)

    grid = TEMPERATURE_GRID if not config.test else (((alphas_c[0], ihs[0]),),)
    postprocess.draw_temperature_grid(config, EXAMPLE_2, grid)


if __name__ == "__main__":
    run_example(main, setup_module)
