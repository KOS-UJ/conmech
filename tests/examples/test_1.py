"""
Created at 21.08.2019

@author: Piotr Bartman
"""

from simulation.simulation_runner import SimulationRunner
from examples.example_1 import Setup


def test():
    setup = Setup()
    SimulationRunner.run(setup)
