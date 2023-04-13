import numpy as np

from conmech.helpers.config import Config, SimulationConfig
from conmech.properties.body_properties import TimeDependentBodyProperties
from conmech.properties.mesh_properties import MeshProperties
from conmech.properties.obstacle_properties import ObstacleProperties
from conmech.properties.schedule import Schedule
from conmech.scenarios.scenarios import (
    M_BALL_3D,
    M_BUNNY_3D_LIFTED,
    M_CUBE_3D,
    M_TWIST_3D,
    Scenario,
    default_body_prop,
    default_obstacle_prop,
    f_rotate_3d,
)
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


def main(mesh_density=16, final_time=2, plot_animation=True):  # 100
    schedule = Schedule(final_time=final_time, time_step=0.01)
    basic_scenarios = [
        Scenario(
            name="cube_throw",
            mesh_prop=MeshProperties(
                dimension=3, mesh_type=M_CUBE_3D, scale=[1], mesh_density=[mesh_density]
            ),
            body_prop=default_body_prop,
            schedule=schedule,
            forces_function=f_rotate_3d,
            obstacle=Obstacle(
                np.array([[[-1.0, 0.0, 1.0]], [[2.0, 0.0, 0.0]]]), default_obstacle_prop
            ),
            simulation_config=simulation_config,
        ),
        Scenario(
            name="ball_roll",
            mesh_prop=MeshProperties(
                dimension=3, mesh_type=M_BALL_3D, scale=[1], mesh_density=[mesh_density]
            ),
            body_prop=default_body_prop,
            schedule=schedule,
            forces_function=np.array([0.0, 0.0, -0.5]),
            obstacle=Obstacle(
                np.array([[[0.3, 0.2, 1.0]], [[0.0, 0.0, -0.01]]]), default_obstacle_prop
            ),
            simulation_config=simulation_config,
        ),
        Scenario(
            name="ball_throw",
            mesh_prop=MeshProperties(
                dimension=3, mesh_type=M_BALL_3D, scale=[1], mesh_density=[mesh_density]
            ),
            body_prop=TimeDependentBodyProperties(
                mu=12,
                lambda_=12,
                theta=4,
                zeta=4,
                mass_density=1.0,
            ),
            schedule=schedule,
            forces_function=f_rotate_3d,
            obstacle=Obstacle(
                np.array([[[0.0, 0.0, 1.0]], [[0.0, 0.0, 0.3]]]), default_obstacle_prop
            ),
            simulation_config=simulation_config,
        ),
        Scenario(
            name="twist_roll",
            mesh_prop=MeshProperties(
                dimension=3,
                mesh_type=M_TWIST_3D,
                scale=[1],
                mesh_density=[mesh_density],
            ),
            body_prop=TimeDependentBodyProperties(
                mu=12,
                lambda_=12,
                theta=4,
                zeta=4,
                mass_density=1.0,
            ),
            schedule=schedule,
            forces_function=np.array([0.0, 0.0, -1.0]),
            obstacle=Obstacle(
                np.array([[[0.0, 0.0, 1.0]], [[0.0, 0.0, 0.1]]]), default_obstacle_prop
            ),
            simulation_config=simulation_config,
        ),
    ]

    advanced_scenarios = [
        Scenario(
            name="bunny_fall",
            mesh_prop=MeshProperties(
                dimension=3,
                mesh_type=M_BUNNY_3D_LIFTED,
                scale=[1],
                mesh_density=[mesh_density],
            ),
            body_prop=TimeDependentBodyProperties(
                mu=8,
                lambda_=8,
                theta=8,
                zeta=8,
                mass_density=1.0,
            ),
            schedule=schedule,
            forces_function=np.array([0.0, 0.0, -1.0]),
            obstacle=Obstacle(
                np.array([[[0.0, 0.7, 1.0]], [[1.0, 1.0, 0.0]]]),
                ObstacleProperties(hardness=100.0, friction=5.0),
                all_mesh=None,
            ),
            simulation_config=simulation_config,
        ),
        Scenario(
            name="bunny_roll",
            mesh_prop=MeshProperties(
                dimension=3,
                mesh_type=M_BUNNY_3D_LIFTED,
                scale=[1],
                mesh_density=[mesh_density],
            ),
            body_prop=TimeDependentBodyProperties(
                mu=12,
                lambda_=12,
                theta=16,
                zeta=16,
                mass_density=1.0,
            ),
            schedule=schedule,
            forces_function=f_rotate_3d,
            obstacle=Obstacle(
                np.array([[[0.0, 0.0, 1.0]], [[0.0, 0.0, 0.3]]]), default_obstacle_prop
            ),
            simulation_config=simulation_config,
        ),
    ]

    return simulation_runner.run_examples(
        all_scenarios=[*advanced_scenarios, *basic_scenarios],
        file=__file__,
        plot_animation=plot_animation,
        config=Config(shell=False, animation_backend="matplotlib"),
    )


if __name__ == "__main__":
    main()
