import numpy as np
import pytest

from conmech.helpers.config import Config
from conmech.properties.mesh_properties import MeshProperties
from conmech.properties.schedule import Schedule
from conmech.scenarios import scenarios
from conmech.scenarios.scenarios import TemperatureScenario
from conmech.simulations.simulation_runner import RunScenarioConfig, run_scenario
from conmech.state.obstacle import Obstacle


def generate_test_suits():
    obstacle = Obstacle(
        np.array([[[0.7, 1.0]], [[0.0, 0.1]]]), scenarios.default_temp_obstacle_prop
    )
    scenario = TemperatureScenario(
        name=f"polygon_temp",
        mesh_prop=MeshProperties(
            dimension=2, mesh_type=scenarios.M_POLYGON, scale=[1], mesh_density=[3]
        ),
        body_prop=scenarios.default_temp_body_prop,
        schedule=Schedule(final_time=1.5),
        forces_function=np.array([1, -1]),
        obstacle=obstacle,
        heat_function=np.array([0]),
    )

    expected_boundary_nodes = np.array(
        [
            [0.54910262, 0.07680929],
            [1.11805496, -0.70722993],
            [2.24000807, 0.11059731],
            [1.1482402, 0.49231939],
            [0.73988253, -0.21451098],
            [0.93197245, -0.47824777],
            [1.497036, -0.469747],
            [1.87060035, -0.19163718],
            [1.87667203, 0.23786271],
            [1.51105671, 0.36491778],
            [0.85102353, 0.28260014],
        ]
    )

    expected_temperature = np.array(
        [
            [0.02525269],
            [0.15827907],
            [0.01420818],
            [0.01189525],
            [0.05136521],
            [0.11858741],
            [0.09876706],
            [0.031813],
            [0.01594183],
            [0.01447565],
            [0.0185746],
            [0.05934548],
            [0.02056129],
            [0.03325895],
            [0.03015383],
        ]
    )

    yield scenario, expected_boundary_nodes, expected_temperature


@pytest.mark.parametrize(
    "scenario, expected_boundary_nodes, expected_temperature",
    list(generate_test_suits()),
)
def test_simulation(scenario, expected_boundary_nodes, expected_temperature):
    config = Config()
    setting, _, _ = run_scenario(
        solve_function=scenario.get_solve_function(),
        scenario=scenario,
        config=config,
        run_config=RunScenarioConfig(
            catalog=f"TEST_{scenario.name}",
            simulate_dirty_data=False,
            plot_animation=False,
        ),
    )

    # print result
    np.set_printoptions(precision=8, suppress=True)
    print(repr(setting.body.state.position.boundary_nodes))
    print(repr(setting.t_old))

    np.testing.assert_array_almost_equal(setting.body.state.position.boundary_nodes, expected_boundary_nodes, decimal=2)
    np.testing.assert_array_almost_equal(setting.t_old, expected_temperature, decimal=2)
