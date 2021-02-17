"""
Created at 21.08.2019
"""

from simulation.simulation_runner import SimulationRunner
from examples.example_static import Setup


def test():
    setup = Setup()
    runner = SimulationRunner(setup)
    _ = runner.run()
