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
            [1.06804534, 0.25261651],
            [0.39525366, 0.83499392],
            [0.25186537, 0.01838707],
            [1.048205, 0.51715704],
            [0.89735532, 0.73470778],
            [0.65934157, 0.85237048],
            [0.17348461, 0.69142589],
            [0.06188244, 0.45692444],
            [0.08755489, 0.20245103],
            [0.46242279, -0.12626741],
            [0.73087265, -0.12743358],
            [0.95546718, 0.01399797],
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
            plot_animation=False,
        ),
    )

    # print result
    np.set_printoptions(precision=8, suppress=True)
    print(repr(setting.body.state.position.boundary_nodes))

    np.testing.assert_allclose(
        setting.body.state.position.boundary_nodes, expected_boundary_nodes, atol=0.03
    )
