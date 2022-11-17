"""
Created at 21.08.2019
"""
from dataclasses import dataclass

import numpy as np
import pytest

from conmech.mesh.boundaries_description import BoundariesDescription
from conmech.scenarios.problems import PoissonProblem
from conmech.simulations.problem_solver import PoissonSolver


@pytest.fixture(params=["direct"])
def solving_method(request):
    return request.param


def generate_test_suits():
    test_suites = []

    # Simple example

    @dataclass()
    class StaticSetup(PoissonProblem):
        grid_height: ... = 1
        elements_number: ... = (4, 4)

        @staticmethod
        def inner_forces(x: np.ndarray) -> np.ndarray:
            return np.array([1000.0])

        @staticmethod
        def outer_forces(x: np.ndarray) -> np.ndarray:
            return np.array([3.0])

        boundaries: ... = BoundariesDescription(dirichlet=lambda x: x[0] == 0 or x[0] == 1)

    setup_1 = StaticSetup(mesh_type="cross")

    expected_temperature_1 = [
        52.44178922,
        115.40502451,
        115.40502451,
        52.44178922,
        52.28737745,
        115.07414216,
        115.07414216,
        52.28737745,
        52.28737745,
        115.07414216,
        115.07414216,
        52.28737745,
        52.44178922,
        115.40502451,
        115.40502451,
        52.44178922,
        94.21875,
        125.65625,
        94.21875,
        94.09742647,
        125.49080882,
        94.09742647,
        94.21875,
        125.65625,
        94.21875,
        94.71507353,
        126.19669118,
        94.71507353,
        94.71507353,
        126.19669118,
        94.71507353,
    ]

    test_suites.append((setup_1, expected_temperature_1))

    # various changes

    @dataclass()
    class StaticSetup(PoissonProblem):
        grid_height: ... = 1.37
        elements_number: ... = (2, 5)

        @staticmethod
        def inner_forces(x):
            return np.array([0])

        @staticmethod
        def outer_forces(x):
            return np.array([100.0])

        boundaries: ... = BoundariesDescription(dirichlet=lambda x: x[0] == 0)

    setup_2 = StaticSetup(mesh_type="cross")
    expected_temperature_vector_2 = [
        1147.375,
        1010.375,
        804.875,
        530.875,
        188.375,
        1147.375,
        1010.375,
        804.875,
        530.875,
        188.375,
        1078.875,
        907.625,
        667.875,
        359.625,
        393.875,
        702.125,
        941.875,
        1113.125,
        1215.875,
        1181.625,
        1215.875,
        1113.125,
        941.875,
        702.125,
        393.875,
    ]

    test_suites.append((setup_2, expected_temperature_vector_2))

    return test_suites


@pytest.mark.parametrize("setup, expected_temperature_vector", generate_test_suits())
def test_direct_solver(solving_method, setup, expected_temperature_vector):
    runner = PoissonSolver(setup, solving_method)
    result = runner.solve()

    temperature = result.temperature
    # print result
    np.set_printoptions(precision=8, suppress=True)

    np.testing.assert_array_almost_equal(temperature, expected_temperature_vector, decimal=3)
