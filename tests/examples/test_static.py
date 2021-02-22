"""
Created at 21.08.2019
"""

from simulation.simulation_runner import SimulationRunner
from examples.example_static import Setup


def test_direct_solver():
    setup = Setup()
    runner = SimulationRunner(setup)
    _ = runner.run(method="direct", verbose=True)


def test_global_optimization_solver():
    setup = Setup()
    runner = SimulationRunner(setup)
    _ = runner.run(method="global optimization")


def test_schur_complement_solver():
    setup = Setup()
    runner = SimulationRunner(setup)
    _ = runner.run(method="schur")
