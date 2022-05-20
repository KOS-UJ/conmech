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
            [1.06792412, 0.21772392],
            [0.43254531, 0.8165968],
            [0.25266147, 0.01888644],
            [1.05839297, 0.4771357],
            [0.92104196, 0.6979237],
            [0.69213966, 0.82068362],
            [0.21223079, 0.68155382],
            [0.09020785, 0.45630717],
            [0.10436248, 0.20817976],
            [0.46734539, -0.13317279],
            [0.72359486, -0.14231155],
            [0.94581391, -0.00998253],
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
