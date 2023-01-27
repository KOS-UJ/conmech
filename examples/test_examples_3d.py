import numpy as np

from conmech.helpers import cmh
from conmech.helpers.config import Config, SimulationConfig
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
    bunny_fall_3d,
    bunny_obstacles,
    default_body_prop,
    default_obstacle_prop,
    f_rotate_3d,
)
from conmech.simulations import simulation_runner
from conmech.state.obstacle import Obstacle


def main():
    cmh.print_jax_configuration()

    simulation_config = SimulationConfig(
        use_normalization=False,
        use_linear_solver=False,
        use_green_strain=True,
        use_nonconvex_friction_law=False,
        use_constant_contact_integral=False,
        use_lhs_preconditioner=False,
        use_pca=False,
    )

    all_scenarios = [
        # bunny_obstacles(mesh_density=16, scale=1, final_time=2),
        bunny_fall_3d(mesh_density=32, scale=1, final_time=2),
        Scenario(
            name="bunny_fall",
            mesh_prop=MeshProperties(
                dimension=3,
                mesh_type=M_BUNNY_3D,
                scale=[1],
                mesh_density=[32],
            ),
            body_prop=TimeDependentBodyProperties(
                mu=12.0,
                lambda_=12.0,
                theta=4.0,
                zeta=4.0,
                mass_density=1.0,
            ),
            schedule=Schedule(final_time=2),
            forces_function=np.array([0.0, 0.0, -1.0]),
            obstacle=Obstacle(  # 0.3
                np.array([[[0.0, 0.7, 1.0]], [[1.0, 1.0, 0.0]]]),
                ObstacleProperties(hardness=100.0, friction=5.0),
            ),
        ),
        Scenario(
            name="bunny_fall",
            mesh_prop=MeshProperties(
                dimension=3,
                mesh_type=M_BUNNY_3D,
                scale=[1],
                mesh_density=[32],
            ),
            body_prop=TimeDependentBodyProperties(
                mu=8,
                lambda_=8,
                theta=8,
                zeta=8,
                mass_density=1.0,
            ),
            schedule=Schedule(final_time=2, time_step=0.01),
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
            simulation_config=simulation_config,
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
        Scenario(
            name="armadillo_fall",
            mesh_prop=MeshProperties(
                dimension=3,
                mesh_type=M_ARMADILLO_3D,
                scale=[1],
                mesh_density=[16],
            ),
            body_prop=TimeDependentBodyProperties(
                # mu=12,
                # lambda_=12,
                # theta=8,
                # zeta=8,
                mu=8,
                lambda_=8,
                theta=4,
                zeta=4,
                # mu=12,
                # lambda_=12,
                # theta=12,
                # zeta=12,
                mass_density=1.0,
            ),
            schedule=Schedule(final_time=3, time_step=0.001),
            forces_function=np.array([0.0, 0.0, -1.0]),
            obstacle=Obstacle(  # 0.3
                np.array([[[0.0, 0.7, 1.0]], [[1.0, 1.0, 0.0]]]),
                ObstacleProperties(hardness=100.0, friction=0.1),  # 5.0),
            ),
            simulation_config=simulation_config,
        ),
    ]

    simulation_runner.run_examples(
        all_scenarios=all_scenarios,
        file=__file__,
        plot_animation=True,
        config=Config(shell=False),
    )


if __name__ == "__main__":
    main()
