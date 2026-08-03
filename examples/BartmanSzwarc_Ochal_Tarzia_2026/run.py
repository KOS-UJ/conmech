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

import pickle
from pathlib import Path
from typing import Optional

import numpy as np

from conmech.helpers.config import Config
from conmech.simulations.problem_solver import PoissonSolver
from conmech.state.state import TemperatureState

from examples.BartmanSzwarc_Ochal_Tarzia_2026 import setup as setup_module
from examples.BartmanSzwarc_Ochal_Tarzia_2026.setup import (
    ALPHAS,
    CONVERGENCE_SEQUENCES,
    IHS,
    TEMPERATURE_GRID,
    StaticPoissonSetup,
    make_slope_contact_law,
)
from examples.common.runner import run_example


def state_path(config, alpha, ih) -> Path:
    return Path(config.outputs_path) / f"alpha_{alpha}_ih_{ih}"


def load_or_simulate(config, alpha, ih, only_ensure=False) -> Optional[TemperatureState]:
    path = state_path(config, alpha, ih)
    if config.force or not path.exists():
        print(f"{config.force=}, {path.exists()=}")
        simulate(config, alpha, ih)
    if only_ensure:
        return None
    with open(path, "rb") as output:
        return pickle.load(output)


def simulate(config, alpha, ih):
    print(f"Simulate {alpha=}, {ih=}")
    setup = StaticPoissonSetup(setup_module.mesh_description(ih))
    solving_method = setup_module.solving_method(alpha)
    if alpha == np.inf:
        # the limit problem has no contact boundary, so it must not keep the
        # contact law the dataclass defaults to: with one present, Direct solves
        # the system with fsolve on a densified matrix instead of one sparse solve
        setup.boundaries = setup_module.limit_boundaries()
        setup.contact_law_2 = None
    else:
        setup.contact_law_2 = make_slope_contact_law(slope=alpha)
    runner = PoissonSolver(setup, solving_method)

    state = runner.solve(verbose=True, method="Powell")

    if config.outputs_path:
        with open(
            f"{config.outputs_path}/alpha_{alpha}_ih_{ih}",
            "wb+",
        ) as output:
            # Workaround
            state.body.dynamics.force.outer.source = None
            state.body.dynamics.force.inner.source = None
            state.body.properties.relaxation = None
            state.setup = None
            state.constitutive_law = None
            pickle.dump(state, output)


def main(config: Config):
    """
    Entrypoint to example.

    To see result of simulation you need to call from python `main(Config().init())`.
    """
    Path(config.outputs_path).mkdir(parents=True, exist_ok=True)

    alphas = ALPHAS if not config.test else ALPHAS[:1]
    ihs = IHS if not config.test else IHS[:1]
    temperature_grid = TEMPERATURE_GRID if not config.test else (((alphas[0], ihs[0]),),)
    convergence_sequences = (
        CONVERGENCE_SEQUENCES
        if not config.test
        else (
            (
                ((alphas[0], ihs[0]),),
                None,
            ),
        )
    )

    for alpha in alphas:
        for ih in ihs:
            print(f"Configuration: {alpha=}, {ih=}")
            load_or_simulate(config, alpha, ih, only_ensure=True)

    all_params = set()
    for item in temperature_grid:
        for alpha, ih in item:
            all_params.add((alpha, ih))
    for row in convergence_sequences:
        for seq in row:
            if seq is None:
                continue
            for alpha, ih in seq:
                all_params.add((alpha, ih))
    for alpha, ih in all_params:
        print(f"Ensuring state: {alpha=}, {ih=}")
        load_or_simulate(config, alpha, ih, only_ensure=True)

    # imported here, not at module level: postprocess use load_or_simulate
    # from this module, TODO: move common parts in one place
    from examples.BartmanSzwarc_Ochal_Tarzia_2026 import postprocess

    postprocess.draw_temperature_grid(config, temperature_grid)
    postprocess.draw_convergence_plots(config, convergence_sequences, ihs, alphas)


if __name__ == "__main__":
    run_example(main, setup_module)
