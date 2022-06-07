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
            [0.97032682, -0.17129993],
            [1.23029796, -0.79432084],
            [2.08475062, -0.41270695],
            [1.4338109, 0.00029222],
            [1.04594543, -0.39300606],
            [1.12906323, -0.59575693],
            [1.51336664, -0.6930302],
            [1.80087124, -0.55805889],
            [1.86914466, -0.27773565],
            [1.65106343, -0.14039698],
            [1.20624354, -0.08759143],
        ]
        # [
        #     [0.89666935, -0.03286421],
        #     [1.14926763, -0.72947624],
        #     [2.16275262, -0.34049936],
        #     [1.42885102, 0.15737338],
        #     [0.98063851, -0.28724665],
        #     [1.05931205, -0.50941969],
        #     [1.49803317, -0.63293531],
        #     [1.83137265, -0.49327925],
        #     [1.9207431, -0.17544424],
        #     [1.67460597, -0.00933442],
        #     [1.1661498, 0.06074662],
        # ]
    )

    expected_temperature = np.array(
        [
            [0.00693524],
            [0.06854819],
            [-0.02738734],
            [-0.01701619],
            [0.01574496],
            [0.04521403],
            [0.02033946],
            [-0.01970288],
            [-0.02703819],
            [-0.02351213],
            [-0.00535789],
            [0.0028299],
            [-0.01613435],
            [-0.01297726],
            [0.00857497],
        ]
        # [
        #     [0.08007115],
        #     [0.17789804],
        #     [0.0421322],
        #     [0.05335039],
        #     [0.09470152],
        #     [0.14607311],
        #     [0.11890882],
        #     [0.05580965],
        #     [0.04377635],
        #     [0.04786408],
        #     [0.06674024],
        #     [0.09342465],
        #     [0.05998566],
        #     [0.06096771],
        #     [0.07763867],
        # ]
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
            plot_animation=config.plot_tests,
        ),
    )

    np.set_printoptions(precision=8, suppress=True)

    np.testing.assert_array_almost_equal(setting.boundary_nodes, expected_boundary_nodes, decimal=2)
    np.testing.assert_array_almost_equal(setting.t_old, expected_temperature, decimal=2)
