from deep_conmech.common import simulation_runner
from deep_conmech.scenarios import *
from deep_conmech.simulator.mesh.mesh_builders_3d import *


def main(mesh_density=3, final_time=1, plot_animation=True):
    all_scenarios = [
        Scenario(
            id="twist_roll",
            mesh_data=MeshData(
                dimension=3, mesh_type=m_twist_3d, scale=[1], mesh_density=[mesh_density]
            ),
            body_prop=default_body_prop,
            obstacle_prop=default_obstacle_prop,
            schedule=Schedule(final_time=final_time),
            forces_function=np.array([0.0, 0.0, -0.5]),
            obstacles=np.array([[[0.3, 0.2, 1.0]], [[0.0, 0.0, -0.01]]]),
        ),
        Scenario(
            id="ball_roll",
            mesh_data=MeshData(
                dimension=3, mesh_type=m_ball_3d, scale=[1], mesh_density=[mesh_density]
            ),
            body_prop=default_body_prop,
            obstacle_prop=default_obstacle_prop,
            schedule=Schedule(final_time=final_time),
            forces_function=np.array([0.0, 0.0, -0.5]),
            obstacles=np.array([[[0.3, 0.2, 1.0]], [[0.0, 0.0, -0.01]]]),
        ),
        Scenario(
            id="ball_throw",
            mesh_data=MeshData(
                dimension=3, mesh_type=m_ball_3d, scale=[1], mesh_density=[mesh_density]
            ),
            body_prop=default_body_prop,
            obstacle_prop=default_obstacle_prop,
            schedule=Schedule(final_time=final_time),
            forces_function=f_rotate_3d,
            obstacles=np.array([[[-1.0, 0.0, 1.0]], [[2.0, 0.0, 0.0]]]),
        ),
        Scenario(
            id="cube_throw",
            mesh_data=MeshData(
                dimension=3, mesh_type=m_cube_3d, scale=[1], mesh_density=[mesh_density]
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
