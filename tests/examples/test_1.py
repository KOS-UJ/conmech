"""
Created at 21.08.2019
"""

from simulation.simulation_runner import SimulationRunner
from examples.example_basic import Setup
from utils.drawer import Drawer


def test():
    setup = Setup()
    runner = SimulationRunner(setup)
    solver = runner.run()
    Drawer(solver).draw()
