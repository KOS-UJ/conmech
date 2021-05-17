"""
Created at 21.08.2019
"""

from conmech.problem_solver import Static as StaticProblem
from examples.example_static import StaticSetup

setup = StaticSetup()
runner = StaticProblem(setup, "direct")


def test_direct_solver():
    runner.solving_method = "direct"
    _ = runner.solve()


def test_global_optimization_solver():
    runner.solving_method = "global optimization"
    _ = runner.solve()


def test_schur_complement_solver():
    runner.solving_method = "schur"
    _ = runner.solve()
