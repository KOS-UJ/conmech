"""
Created at 21.08.2019
"""

from simulation.simulation_runner import SimulationRunner
from examples.example_static import Setup

setup = Setup()
runner = SimulationRunner(setup)


def test_direct_solver():
    _ = runner.run(method="direct", verbose=True)


def test_global_optimization_solver():
    _ = runner.run(method="global optimization")


def test_schur_complement_solver():
    _ = runner.run(method="schur")
