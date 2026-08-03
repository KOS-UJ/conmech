# CONMECH @ Jagiellonian University in Kraków
#
# Copyright (C) 2022-2026  Piotr Bartman <piotr.bartman@uj.edu.pl>
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
from conmech.simulations.problem_solver import PoissonSolver

from examples.common.runner import run_example
from examples.example_poisson import postprocess, setup as setup_module


def simulate():
    setup = setup_module.StaticPoissonSetup(setup_module.mesh_description())
    runner = PoissonSolver(setup, setup_module.SOLVING_METHOD)
    return runner.solve(verbose=True)


def main(config: Config):
    postprocess.draw(simulate(), config)


if __name__ == "__main__":
    run_example(main, setup_module)
