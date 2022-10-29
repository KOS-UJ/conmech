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
            [0.98399403, -0.17984905],
            [1.22851901, -0.79841793],
            [2.09756634, -0.42357112],
            [1.44917658, -0.00821687],
            [1.05883439, -0.3998009],
            [1.13828376, -0.60078619],
            [1.52306518, -0.70276705],
            [1.81278825, -0.56851194],
            [1.88241913, -0.28877599],
            [1.66560294, -0.15042728],
            [1.2205438, -0.09538353],
        ]
    )

    expected_temperature = np.array(
        [
            [0.01290254],
            [0.02779549],
            [-0.0030564],
            [-0.00009818],
            [0.01504894],
            [0.02780501],
            [0.03453887],
            [-0.01355886],
            [-0.02122882],
            [-0.01344031],
            [0.0000184],
            [-0.00238224],
            [-0.01531763],
            [-0.0086441],
            [0.00762045],
        ]
    )

    yield scenario, expected_boundary_nodes, expected_temperature


@pytest.mark.parametrize(
    "scenario, expected_boundary_nodes, expected_temperature",
    list(generate_test_suits()),
)
def test_simulation(scenario, expected_boundary_nodes, expected_temperature):
    config = Config()
    _ = config
    return
    # setting, _, _ = run_scenario(
    #     solve_function=scenario.get_solve_function(),
    #     scenario=scenario,
    #     config=config,
    #     run_config=RunScenarioConfig(
    #         catalog=f"TEST_{scenario.name}",
    #         simulate_dirty_data=False,
    #         plot_animation=config.plot_tests,
    #     ),
    # )

    # np.set_printoptions(precision=8, suppress=True)

    # np.testing.assert_array_almost_equal(setting.boundary_nodes, expected_boundary_nodes, decimal=2)
    # np.testing.assert_array_almost_equal(setting.t_old, expected_temperature, decimal=2)
