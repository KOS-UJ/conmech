"""
Created at 21.08.2019
"""

from conmech.problem_solver import Dynamic as DynamicProblem
from examples.example_dynamic import DynamicSetup

setup = DynamicSetup()
runner = DynamicProblem(setup, "direct")


def test_direct_solver():
    runner.solving_method = "direct"
    _ = runner.solve(n_steps=8)


def test_global_optimization_solver():
    runner.solving_method = "global optimization"
    _ = runner.solve(n_steps=8)


def test_schur_complement_solver():
    runner.solving_method = "schur"
    _ = runner.solve(n_steps=8)
