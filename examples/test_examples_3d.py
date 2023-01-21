import numpy as np

from conmech.helpers.config import Config
from conmech.properties.body_properties import TimeDependentBodyProperties
from conmech.properties.mesh_properties import MeshProperties
from conmech.properties.obstacle_properties import ObstacleProperties
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
    f_rotate_3d,
)
from conmech.simulations import simulation_runner
from conmech.state.obstacle import Obstacle


def main():
    all_scenarios = [
        Scenario(
            name="bunny_fall",
            mesh_prop=MeshProperties(
                dimension=3,
                mesh_type=M_BUNNY_3D,
                scale=[1],
                mesh_density=[16],  # 32
            ),
            body_prop=TimeDependentBodyProperties(
                mu=8,
                lambda_=8,
                theta=8,
                zeta=8,
                mass_density=1.0,
            ),
            schedule=Schedule(final_time=2),
            forces_function=np.array([0.0, 0.0, -1.0]),
            obstacle=Obstacle(
                None,
                ObstacleProperties(hardness=100.0, friction=5.0),
                all_mesh=[
                    MeshProperties(
                        dimension=3,
                        mesh_type="slide_left",
                        scale=[1],
                        mesh_density=[16],
                        initial_position=[0, 0, -0.5],
                    ),
                ],  # x,y,z front,right,bottom
            ),
        ),
        # Scenario(
        #     name="bunny_roll",
        #     mesh_prop=MeshProperties(
        #         dimension=3,
        #         mesh_type=M_BUNNY_3D,
        #         scale=[1],
        #         mesh_density=[16],  # 8
        #     ),
        #     body_prop=TimeDependentBodyProperties(
        #         mu=12,
        #         lambda_=12,
        #         theta=16,
        #         zeta=16,
        #         mass_density=1.0,
        #     ),
        #     schedule=Schedule(final_time=final_time),
        #     forces_function=f_rotate_3d,
        #     obstacle=Obstacle(
        #         np.array([[[0.0, 0.0, 1.0]], [[0.0, 0.0, 0.3]]]), default_obstacle_prop
        #     ),
        # )
        # Scenario(
        #     name="armadillo_roll",
        #     mesh_prop=MeshProperties(
        #         dimension=3,
        #         mesh_type=M_ARMADILLO_3D,
        #         scale=[1],
        #         mesh_density=[mesh_density],
        #     ),
        #     body_prop=TimeDependentBodyProperties(
        #         mu=12,
        #         lambda_=12,
        #         theta=8,
        #         zeta=8,
        #         mass_density=1.0,
        #     ),
        #     schedule=Schedule(final_time=final_time),
        #     forces_function=np.array([0.0, 0.0, -1.0]),
        #     obstacle=Obstacle(  # 0.3
        #         np.array([[[0.0, 0.7, 1.0]], [[1.0, 1.0, 0.0]]]),
        #         ObstacleProperties(hardness=100.0, friction=5.0),
        #     ),
        # ),
    ]

    simulation_runner.run_examples(
        all_scenarios=all_scenarios,
        file=__file__,
        plot_animation=True,
        config=Config(shell=False),
    )


if __name__ == "__main__":
    main()
