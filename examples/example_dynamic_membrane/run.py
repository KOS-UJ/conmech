# CONMECH @ Jagiellonian University in Kraków
#
# Copyright (C) 2023-2026  Piotr Bartman-Szwarc <piotr.bartman@uj.edu.pl>
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
from conmech.helpers.config import Config
from conmech.simulations.problem_solver import WaveSolver

from examples.common.runner import run_example
from examples.example_dynamic_membrane import postprocess, setup as setup_module


def simulate(test: bool = False):
    setup = setup_module.MembraneSetup(setup_module.mesh_description(test))
    runner = WaveSolver(setup, setup_module.SOLVING_METHOD)
    n_steps = setup_module.N_STEPS_TEST if test else setup_module.N_STEPS
    return runner.solve(
        n_steps=n_steps,
        output_step=(0, n_steps),
        initial_displacement=setup.initial_displacement,
        initial_velocity=setup.initial_velocity,
        verbose=True,
    )


def main(config: Config):
    postprocess.draw(simulate(test=config.test), config)


if __name__ == "__main__":
    run_example(main, setup_module)
