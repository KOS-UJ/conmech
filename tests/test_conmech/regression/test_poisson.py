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

    @dataclass()
    class StaticSetup(PoissonProblem):
        grid_height: ... = 1
        elements_number: ... = (4, 4)

        @staticmethod
        def inner_forces(x: np.ndarray, t: float) -> np.ndarray:
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
        0,
        0,
        0,
        0,
        0,
        0,
        0,
        0,
        0,
        0,
    ]

    test_suites.append((setup_1, expected_temperature_1))

    # various changes

    @dataclass()
    class StaticSetup(PoissonProblem):
        grid_height: ... = 1.37
        elements_number: ... = (3, 5)

        @staticmethod
        def inner_forces(x):
            return np.array([0])

        @staticmethod
        def outer_forces(x):
            return np.array([10.0])

        boundaries: ... = BoundariesDescription(
            dirichlet=(lambda x: x[0] == 0, lambda x: np.full_like(x[:, 0], 5))
        )

    setup_2 = StaticSetup(mesh_type="cross")
    expected_temperature_vector_2 = [
        63.35184411,
        55.74064007,
        45.0837835,
        31.36566229,
        14.36882193,
        61.82964512,
        54.21871987,
        43.56576633,
        29.90200875,
        13.66235614,
        63.35184411,
        55.74064007,
        45.0837835,
        31.36566229,
        14.36882193,
        58.78521229,
        49.65222744,
        37.47930522,
        22.32471228,
        58.78521229,
        49.65222744,
        37.47930522,
        22.32471228,
        25.15057545,
        40.50805623,
        52.69554512,
        61.82957542,
        67.91851077,
        64.87407795,
        64.87407795,
        67.91851077,
        61.82957542,
        52.69554512,
        40.50805623,
        25.15057545,
        5.0,
        5.0,
        5.0,
        5.0,
    ]

    test_suites.append((setup_2, expected_temperature_vector_2))

    return test_suites


@pytest.mark.parametrize("setup, expected_temperature_vector", generate_test_suits())
def test_direct_solver(solving_method, setup, expected_temperature_vector):
    runner = PoissonSolver(setup, solving_method)
    result = runner.solve()

    temperature = result.temperature

    np.set_printoptions(precision=8, suppress=True)

    np.testing.assert_array_almost_equal(temperature, expected_temperature_vector, decimal=3)
