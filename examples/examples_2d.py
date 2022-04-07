import numpy as np

import conmech.scenarios.scenarios as scenarios
from conmech.helpers.config import Config
from conmech.properties.mesh_properties import MeshProperties
from conmech.properties.obstacle_properties import ObstacleProperties
from conmech.properties.schedule import Schedule
from conmech.scenarios.scenarios import Scenario
from conmech.simulations import simulation_runner


def main(mesh_density=3, final_time=5, plot_animation=True):
    config = Config(SHELL=False)
    all_scenarios = [
        Scenario(
            id="circle_slide_roll",
            mesh_data=MeshProperties(
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
            mesh_data=MeshProperties(
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
            mesh_data=MeshProperties(
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
            mesh_data=MeshProperties(
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
            mesh_data=MeshProperties(
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
