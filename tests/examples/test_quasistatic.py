"""
Created at 21.08.2019
"""

from conmech.problem_solver import Quasistatic as QuasistaticProblem
from examples.example_quasistatic import QuasistaticSetup

setup = QuasistaticSetup()
runner = QuasistaticProblem(setup, "direct")


def test_direct_solver():
    runner.solving_method = "direct"
    _ = runner.solve(n_steps=8)


def test_global_optimization_solver():
    runner.solving_method = "global optimization"
    _ = runner.solve(n_steps=8)


def test_schur_complement_solver():
    runner.solving_method = "schur"
    _ = runner.solve(n_steps=8)
