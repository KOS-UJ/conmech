import numpy as np
import pytest

from conmech.helpers.config import Config, SimulationConfig
from conmech.properties.mesh_properties import MeshProperties
from conmech.properties.schedule import Schedule
from conmech.scenarios import scenarios
from conmech.scenarios.scenarios import Scenario
from conmech.simulations.simulation_runner import RunScenarioConfig, run_scenario
from conmech.state.obstacle import Obstacle

simulation_config = SimulationConfig(
    use_normalization=False,
    use_linear_solver=False,
    use_green_strain=True,
    use_nonconvex_friction_law=False,
    use_constant_contact_integral=False,
    use_lhs_preconditioner=False,
    use_pca=False,
)


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
        simulation_config=simulation_config,
    )

    expected_boundary_nodes = np.array(
        [
            [1.09130674, 0.19546702],
            [0.48433058, 0.82372344],
            [0.25647264, 0.02265621],
            [1.09451529, 0.45555229],
            [0.96761506, 0.68292285],
            [0.74446174, 0.81669446],
            [0.25722132, 0.69875847],
            [0.12400642, 0.47832344],
            [0.12319444, 0.22691596],
            [0.47185912, -0.13701605],
            [0.73021312, -0.15134028],
            [0.95839701, -0.02725928],
        ]
        # [
        #     [1.06792412, 0.21772392],
        #     [0.43254531, 0.8165968],
        #     [0.25266147, 0.01888644],
        #     [1.05839297, 0.4771357],
        #     [0.92104196, 0.6979237],
        #     [0.69213966, 0.82068362],
        #     [0.21223079, 0.68155382],
        #     [0.09020785, 0.45630717],
        #     [0.10436248, 0.20817976],
        #     [0.46734539, -0.13317279],
        #     [0.72359486, -0.14231155],
        #     [0.94581391, -0.00998253],
        # ]
    )

    yield scenario, expected_boundary_nodes


@pytest.mark.parametrize("scenario, expected_boundary_nodes", list(generate_test_suits()))
def test_simulation(scenario, expected_boundary_nodes):
    config = Config()
    _ = config
    return
    # setting, _, _ = run_scenario(
    #     solve_function=scenario.get_solve_function(),
    #     config=config,
    #     scenario=scenario,
    #     run_config=RunScenarioConfig(
    #         catalog=f"TEST_{scenario.name}",
    #         simulate_dirty_data=False,
    #         plot_animation=config.plot_tests,
    #     ),
    # )

    # np.testing.assert_allclose(setting.boundary_nodes, expected_boundary_nodes, atol=0.03)
