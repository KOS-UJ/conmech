import numpy as np

from conmech.helpers.config import Config
from conmech.properties.mesh_properties import MeshProperties
from conmech.properties.schedule import Schedule
from conmech.scenarios.scenarios import Scenario, default_body_prop, default_obstacle_prop, \
    f_rotate_3d, \
    M_BALL_3D, M_TWIST_3D, M_CUBE_3D
from conmech.simulations import simulation_runner


def main(mesh_density=3, final_time=1, plot_animation=True):
    all_scenarios = [
        Scenario(
            name="twist_roll",
            mesh_data=MeshProperties(
                dimension=3, mesh_type=M_TWIST_3D, scale=[1], mesh_density=[mesh_density]
            ),
            body_prop=default_body_prop,
            obstacle_prop=default_obstacle_prop,
            schedule=Schedule(final_time=final_time),
            forces_function=np.array([0.0, 0.0, -0.5]),
            obstacles=np.array([[[0.3, 0.2, 1.0]], [[0.0, 0.0, -0.01]]]),
        ),
        Scenario(
            name="ball_roll",
            mesh_data=MeshProperties(
                dimension=3, mesh_type=M_BALL_3D, scale=[1], mesh_density=[mesh_density]
            ),
            body_prop=default_body_prop,
            obstacle_prop=default_obstacle_prop,
            schedule=Schedule(final_time=final_time),
            forces_function=np.array([0.0, 0.0, -0.5]),
            obstacles=np.array([[[0.3, 0.2, 1.0]], [[0.0, 0.0, -0.01]]]),
        ),
        Scenario(
            name="ball_throw",
            mesh_data=MeshProperties(
                dimension=3, mesh_type=M_BALL_3D, scale=[1], mesh_density=[mesh_density]
            ),
            body_prop=default_body_prop,
            obstacle_prop=default_obstacle_prop,
            schedule=Schedule(final_time=final_time),
            forces_function=f_rotate_3d,
            obstacles=np.array([[[-1.0, 0.0, 1.0]], [[2.0, 0.0, 0.0]]]),
        ),
        Scenario(
            name="cube_throw",
            mesh_data=MeshProperties(
                dimension=3, mesh_type=M_CUBE_3D, scale=[1], mesh_density=[mesh_density]
            ),
            body_prop=default_body_prop,
            obstacle_prop=default_obstacle_prop,
            schedule=Schedule(final_time=final_time),
            forces_function=f_rotate_3d,
            obstacles=np.array([[[-1.0, 0.0, 1.0]], [[2.0, 0.0, 0.0]]]),
        ),
    ]

    simulation_runner.run_examples(
        all_scenarios=all_scenarios,
        file=__file__,
        plot_animation=plot_animation,
        config=Config(SHELL=False)
    )


if __name__ == "__main__":
    main()
