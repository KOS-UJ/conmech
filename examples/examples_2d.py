import numpy as np

import conmech.scenarios.scenarios as scenarios
from conmech.helpers.config import Config, SimulationConfig
from conmech.properties.mesh_properties import MeshProperties
from conmech.properties.obstacle_properties import ObstacleProperties
from conmech.properties.schedule import Schedule
from conmech.scenarios.scenarios import Scenario
from conmech.simulations import simulation_runner
from conmech.state.obstacle import Obstacle

simulation_config = SimulationConfig(
    use_normalization=False,
    use_linear_solver=False,
    use_green_strain=True,
    use_nonconvex_friction_law=False,
    use_constant_contact_integral=False,
    use_lhs_preconditioner=False,
    with_self_collisions=False,
    use_pca=False,
)


def main(mesh_density=20, final_time=5, plot_animation=True):  # 40
    schedule = Schedule(final_time=final_time, time_step=0.01)
    config = Config(shell=False)
    obstacles = [
        Obstacle(
            np.array([[[0.7, 1.0], [-0.3, 1.0]], [[0.0, -0.01], [4.0, -0.01]]]),
            scenarios.default_obstacle_prop,
        ),
        Obstacle(
            np.array([[[0.0, 1.0]], [[0.0, 0.0]]]),
            ObstacleProperties(hardness=100.0, friction=5.0),
        ),
        Obstacle(
            np.array([[[0.0, 1.0]], [[0.0, 0.0]]]),
            ObstacleProperties(hardness=10.0, friction=5.0),
        ),
        Obstacle(
            np.array([[[0.0, 1.0]], [[0.0, 0.0]]]),
            ObstacleProperties(hardness=100.0, friction=0.5),
        ),
        Obstacle(np.array([[[0.0, 1.0]], [[0.0, 0.0]]]), scenarios.default_obstacle_prop),
    ]
    all_scenarios = [
        scenarios.polygon_mesh_obstacles(
            mesh_density=mesh_density,
            scale=1,
            final_time=final_time,
            simulation_config=simulation_config,
        ),
        Scenario(
            name="circle_slide_roll",
            mesh_prop=MeshProperties(
                dimension=2,
                mesh_type=scenarios.M_CIRCLE,
                scale=[1],
                mesh_density=[mesh_density],
            ),
            body_prop=scenarios.default_body_prop,
            schedule=schedule,
            forces_function=np.array([0.0, -0.5]),
            obstacle=obstacles[0],
            simulation_config=simulation_config,
        ),
        Scenario(
            name="circle_flat_A_roll",
            mesh_prop=MeshProperties(
                dimension=2,
                mesh_type=scenarios.M_CIRCLE,
                scale=[1],
                mesh_density=[mesh_density],
            ),
            body_prop=scenarios.default_body_prop,
            schedule=schedule,
            forces_function=np.array([2.0, -0.5]),
            obstacle=obstacles[1],
            simulation_config=simulation_config,
        ),
        Scenario(
            name="circle_flat_B_roll",
            mesh_prop=MeshProperties(
                dimension=2,
                mesh_type=scenarios.M_CIRCLE,
                scale=[1],
                mesh_density=[mesh_density],
            ),
            body_prop=scenarios.default_body_prop,
            schedule=schedule,
            forces_function=np.array([2.0, -0.5]),
            obstacle=obstacles[2],
            simulation_config=simulation_config,
        ),
        Scenario(
            name="circle_flat_C_roll",
            mesh_prop=MeshProperties(
                dimension=2,
                mesh_type=scenarios.M_CIRCLE,
                scale=[1],
                mesh_density=[mesh_density],
            ),
            body_prop=scenarios.default_body_prop,
            schedule=schedule,
            forces_function=np.array([2.0, -0.5]),
            obstacle=obstacles[3],
            simulation_config=simulation_config,
        ),
        Scenario(
            name="rectangle_flat_roll",
            mesh_prop=MeshProperties(
                dimension=2,
                mesh_type=scenarios.M_RECTANGLE,
                scale=[1],
                mesh_density=[mesh_density],
            ),
            body_prop=scenarios.default_body_prop,
            schedule=schedule,
            forces_function=np.array([2.0, -0.5]),
            obstacle=obstacles[4],
            simulation_config=simulation_config,
        ),
    ]

    return simulation_runner.run_examples(
        all_scenarios=all_scenarios,
        file=__file__,
        plot_animation=plot_animation,
        config=config,
    )


if __name__ == "__main__":
    main()
