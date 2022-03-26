from deep_conmech.common.plotter import plotter_mapper
from deep_conmech.graph.setting.setting_randomized import SettingRandomized
from deep_conmech.scenarios import *
from deep_conmech.simulator.mesh.mesh_builders_3d import *
from deep_conmech.simulator.solver import Solver


def main(mesh_density=3, final_time=3, plot_animation=True):
    all_scenarios = [
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

    for scenario in all_scenarios:
        print("-----")
        plotter_mapper.print_one_dynamic(
            Solver.solve,
            scenario,
            SettingRandomized.get_setting,
            catalog="EXAMPLES 3D",
            simulate_dirty_data=False,
            plot_base=False,
            plot_detailed=True,
            plot_animation=plot_animation
        )


if __name__ == "__main__":
    main()
