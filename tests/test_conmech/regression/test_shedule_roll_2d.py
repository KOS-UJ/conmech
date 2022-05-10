import numpy as np
import pytest

from conmech.helpers.config import Config
from conmech.properties.mesh_properties import MeshProperties
from conmech.properties.schedule import Schedule
from conmech.scenarios import scenarios
from conmech.scenarios.scenarios import Scenario
from conmech.simulations.simulation_runner import RunScenarioConfig, run_scenario
from conmech.state.obstacle import Obstacle


def generate_test_suits():
    scenario = Scenario(
        name="circle_slide_roll",
        mesh_prop=MeshProperties(
            dimension=2,
            mesh_type=scenarios.M_CIRCLE,
            scale=[1],
            mesh_density=[3],
        ),
        body_prop=scenarios.default_body_prop,
        schedule=Schedule(final_time=1.5),
        forces_function=np.array([0.0, -0.5]),
        obstacle=Obstacle(np.array([[[0.7, 1.0]], [[0.0, 0.2]]]), scenarios.default_obstacle_prop),
    )

    expected_boundary_nodes = np.array(
        [
            [1.06836935, 0.21640903],
            [0.43501884, 0.817377],
            [0.2546197, 0.02082238],
            [1.05973179, 0.47571151],
            [0.92315777, 0.69700249],
            [0.69467413, 0.82051599],
            [0.21427316, 0.68286631],
            [0.09160728, 0.45794447],
            [0.10574119, 0.2099006],
            [0.4668943, -0.13030448],
            [0.72252617, -0.14164865],
            [0.94529081, -0.01071903],
        ]
    )

    yield scenario, expected_boundary_nodes


@pytest.mark.parametrize("scenario, expected_boundary_nodes", list(generate_test_suits()))
def test_simulation(scenario, expected_boundary_nodes):
    config = Config()
    setting, _, _ = run_scenario(
        solve_function=scenario.get_solve_function(),
        config=config,
        scenario=scenario,
        run_config=RunScenarioConfig(
            catalog=f"TEST_{scenario.name}",
            simulate_dirty_data=False,
            plot_animation=config.plot_tests,
        ),
    )

    np.testing.assert_array_almost_equal(setting.boundary_nodes, expected_boundary_nodes, decimal=2)
