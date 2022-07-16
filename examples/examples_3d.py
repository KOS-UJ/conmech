import os

os.environ["JAX_ENABLE_X64"] = "1"
# os.environ["JAX_PLATFORM_NAME"] = "cpu"

import numpy as np

from conmech.helpers.config import Config
from conmech.properties.body_properties import DynamicBodyProperties
from conmech.properties.mesh_properties import MeshProperties
from conmech.properties.schedule import Schedule
from conmech.scenarios.scenarios import (
    M_ARMADILLO_3D,
    M_BALL_3D,
    M_BUNNY_3D,
    M_CUBE_3D,
    M_TWIST_3D,
    Scenario,
    default_body_prop,
    default_obstacle_prop,
    f_push_3d,
    f_rotate_3d,
)
from conmech.simulations import simulation_runner
from conmech.state.obstacle import Obstacle


def main(mesh_density=16, final_time=8.0, plot_animation=True):  # 100
    obstacles = [
        Obstacle(np.array([[[0.3, 0.2, 1.0]], [[0.0, 0.0, -0.01]]]), default_obstacle_prop),
        Obstacle(np.array([[[0.3, 0.2, 1.0]], [[0.0, 0.0, -0.01]]]), default_obstacle_prop),
        Obstacle(np.array([[[-1.0, 0.0, 1.0]], [[2.0, 0.0, 0.0]]]), default_obstacle_prop),
        Obstacle(np.array([[[-1.0, 0.0, 1.0]], [[2.0, 0.0, 0.0]]]), default_obstacle_prop),
    ]
    all_scenarios = [
        Scenario(
            name="bunny_roll",
            mesh_prop=MeshProperties(
                dimension=3,
                mesh_type=M_BUNNY_3D,
                scale=[1],
                mesh_density=[mesh_density],
            ),
            body_prop=DynamicBodyProperties(
                mu=12,
                lambda_=12,
                theta=16,
                zeta=16,
                mass_density=1.0,
            ),
            schedule=Schedule(final_time=final_time),
            forces_function=f_rotate_3d,  # np.array([0.0, 0.0, -1.0]),  # f_rotate_3d,  #
            obstacle=Obstacle(
                np.array([[[0.0, 0.0, 1.0]], [[0.0, 0.0, 0.3]]]), default_obstacle_prop
            ),
        ),
        # Scenario(
        #     name="armadillo_roll",
        #     mesh_prop=MeshProperties(
        #         dimension=3,
        #         mesh_type=M_ARMADILLO_3D,
        #         scale=[1],
        #         mesh_density=[mesh_density],
        #     ),
        #     body_prop=DynamicBodyProperties(
        #         mu=8,
        #         lambda_=8,
        #         theta=16,
        #         zeta=16,
        #         mass_density=1.0,
        #     ),
        #     schedule=Schedule(final_time=final_time),
        #     forces_function=np.array([0.0, 0.0, -1.0]),
        #     obstacle=Obstacle(
        #         np.array([[[0.0, 0.0, 1.0]], [[0.0, 0.0, 0.0]]]), default_obstacle_prop
        #     ),
        # ),
        # Scenario(
        #     name="ball_roll",
        #     mesh_prop=MeshProperties(
        #         dimension=3, mesh_type=M_BALL_3D, scale=[1], mesh_density=[mesh_density]
        #     ),
        #     body_prop=default_body_prop,
        #     schedule=Schedule(final_time=final_time),
        #     forces_function=np.array([0.0, 0.0, -0.5]),
        #     obstacle=obstacles[1],
        # ),
        Scenario(
            name="ball_throw",
            mesh_prop=MeshProperties(
                dimension=3, mesh_type=M_BALL_3D, scale=[1], mesh_density=[mesh_density]
            ),
            body_prop=DynamicBodyProperties(
                mu=12,  # 8,
                lambda_=12,  # 8,
                theta=4,
                zeta=4,
                mass_density=1.0,
            ),
            schedule=Schedule(final_time=final_time),
            forces_function=f_rotate_3d,
            obstacle=obstacles[2],
        ),
        # Scenario(
        #     name="cube_throw",
        #     mesh_prop=MeshProperties(
        #         dimension=3, mesh_type=M_CUBE_3D, scale=[1], mesh_density=[mesh_density]
        #     ),
        #     body_prop=default_body_prop,
        #     schedule=Schedule(final_time=final_time),
        #     forces_function=f_rotate_3d,
        #     obstacle=obstacles[3],
        # ),
        Scenario(
            name="twist_roll",
            mesh_prop=MeshProperties(
                dimension=3,
                mesh_type=M_TWIST_3D,
                scale=[1],
                mesh_density=[mesh_density],
            ),
            body_prop=DynamicBodyProperties(
                mu=12,  # 8,
                lambda_=12,  # 8,
                theta=4,
                zeta=4,
                mass_density=1.0,
            ),
            schedule=Schedule(final_time=final_time),
            forces_function=np.array([0.0, 0.0, -1.0]),  # f_rotate_3d,  #
            obstacle=Obstacle(
                np.array([[[0.0, 0.0, 1.0]], [[0.0, 0.0, 0.1]]]), default_obstacle_prop
            ),
        ),
    ]

    simulation_runner.run_examples(
        all_scenarios=all_scenarios,
        file=__file__,
        plot_animation=plot_animation,
        config=Config(shell=False),
    )


if __name__ == "__main__":
    # os.environ["CUDA_VISIBLE_DEVICES"] = "-1"
    os.environ["CUDA_VISIBLE_DEVICES"] = "1"
    # os.environ["XLA_PYTHON_CLIENT_PREALLOCATE"] = "false"
    main()
