import numpy as np

import deep_conmech.scenarios as scenarios
from deep_conmech.common import simulation_runner
from deep_conmech.scenarios import Config, Scenario, MeshData, Schedule, ObstacleProperties


def main(mesh_density=3, final_time=10, plot_animation=True):
    config = Config(SHELL=False)
    all_scenarios = [
        Scenario(
            id="circle_slide_roll",
            mesh_data=MeshData(
                dimension=2,
                mesh_type=scenarios.m_circle,
                scale=[1],
                mesh_density=[mesh_density],
            ),
            body_prop=scenarios.default_body_prop,
            obstacle_prop=scenarios.default_obstacle_prop,
            schedule=Schedule(final_time=final_time),
            forces_function=np.array([0.0, -0.5]),
            obstacles=np.array(
                [[[0.7, 1.0], [-0.3, 1.0]], [[0.0, -0.01], [4.0, -0.01]]]
            ),
        ),
        Scenario(
            id="circle_flat_A_roll",
            mesh_data=MeshData(
                dimension=2,
                mesh_type=scenarios.m_circle,
                scale=[1],
                mesh_density=[mesh_density],
            ),
            body_prop=scenarios.default_body_prop,
            obstacle_prop=ObstacleProperties(hardness=100.0, friction=5.0),
            schedule=Schedule(final_time=final_time),
            forces_function=np.array([2.0, -0.5]),
            obstacles=np.array([[[0.0, 1.0]], [[0.0, 0.0]]]),
        ),
        Scenario(
            id="circle_flat_B_roll",
            mesh_data=MeshData(
                dimension=2,
                mesh_type=scenarios.m_circle,
                scale=[1],
                mesh_density=[mesh_density],
            ),
            body_prop=scenarios.default_body_prop,
            obstacle_prop=ObstacleProperties(hardness=10.0, friction=5.0),
            schedule=Schedule(final_time=final_time),
            forces_function=np.array([2.0, -0.5]),
            obstacles=np.array([[[0.0, 1.0]], [[0.0, 0.0]]]),
        ),
        Scenario(
            id="circle_flat_C_roll",
            mesh_data=MeshData(
                dimension=2,
                mesh_type=scenarios.m_circle,
                scale=[1],
                mesh_density=[mesh_density],
            ),
            body_prop=scenarios.default_body_prop,
            obstacle_prop=ObstacleProperties(hardness=100.0, friction=0.5),
            schedule=Schedule(final_time=final_time),
            forces_function=np.array([2.0, -0.5]),
            obstacles=np.array([[[0.0, 1.0]], [[0.0, 0.0]]]),
        ),
        Scenario(
            id="rectangle_flat_roll",
            mesh_data=MeshData(
                dimension=2,
                mesh_type=scenarios.m_rectangle,
                scale=[1],
                mesh_density=[mesh_density],
            ),
            body_prop=scenarios.default_body_prop,
            obstacle_prop=scenarios.default_obstacle_prop,
            schedule=Schedule(final_time=final_time),
            forces_function=np.array([2.0, -0.5]),
            obstacles=np.array([[[0.0, 1.0]], [[0.0, 0.0]]]),
        ),
    ]

    simulation_runner.run_examples(
        all_scenarios=all_scenarios,
        file=__file__,
        plot_animation=plot_animation,
        config=config
    )


if __name__ == "__main__":
    main()
