"""
Created at 21.08.2019
"""
from dataclasses import dataclass

import numpy as np
import pytest

from conmech.mesh.boundaries_description import BoundariesDescription
from conmech.scenarios.problems import PoissonProblem
from conmech.simulations.problem_solver import PoissonSolver
from tests.test_conmech.regression.std_boundary import standard_boundary_nodes


@pytest.fixture(params=["direct", "global optimization", "schur"])
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
            return np.array([1000.])

        @staticmethod
        def outer_forces(x: np.ndarray) -> np.ndarray:
            return np.array([3.])

        boundaries: ... = BoundariesDescription(
            dirichlet=lambda x: x[0] == 0 or x[0] == 1
        )

    setup_1 = StaticSetup(mesh_type="cross")

    expected_temperature_1 = [57.94791667, 121.32291667, 121.32291667, 57.94791667,
                              57.19791667, 119.57291667, 119.57291667, 57.19791667,
                              57.19791667, 119.57291667, 119.57291667, 57.19791667,
                              57.94791667, 121.32291667, 121.32291667, 57.94791667,
                              93.1875, 124.25, 93.1875, 93.9375,
                              125.25, 93.9375, 93.1875, 124.25,
                              93.1875, 96.9375, 129.25, 96.9375,
                              96.9375, 129.25, 96.9375]

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
            return np.array([100.])

        boundaries: ... = BoundariesDescription(
            dirichlet=lambda x: x[0] == 0
        )

    setup_2 = StaticSetup(mesh_type="cross")
    expected_temperature_vector_2 = [
        [0.0, 0.0],
        [-0.02154956, 0.01364313],
        [-0.04849654, 0.05059958],
        [-0.07590132, 0.0972985],
        [-0.09873572, 0.15498692],
        [-0.12252541, 0.22719522],
        [-0.19937449, 0.26118308],
        [-0.30552747, 0.28092124],
        [-0.27474735, 0.1939756],
        [-0.22880436, 0.13188258],
        [-0.17312159, 0.08296667],
        [-0.10282189, 0.04289061],
        [0.0, 0.0],
        [0.0, 0.0],
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
    print(repr(temperature))

    np.testing.assert_array_almost_equal(
        temperature, expected_temperature_vector, decimal=3
    )
