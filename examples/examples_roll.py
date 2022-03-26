import deep_conmech.scenarios as scenarios
from conmech.helpers import cmh
from deep_conmech.common.plotter import plotter_mapper
from deep_conmech.graph.setting.setting_randomized import SettingRandomized
from deep_conmech.scenarios import *
from deep_conmech.simulator.solver import Solver


def main(mesh_density=5, final_time=5, plot_animation=True):
    all_scenarios = [
        Scenario(
            id="rectangle_small",
            mesh_data=MeshData(
                dimension=2,
                mesh_type=scenarios.m_rectangle,
                scale=[1],
                mesh_density=[4],
            ),
            body_prop=scenarios.default_body_prop,
            obstacle_prop=scenarios.default_obstacle_prop,
            schedule=Schedule(final_time=final_time),
            forces_function=np.array([2.0, -0.5]),
            obstacles=np.array([[[0.0, 1.0]], [[0.0, 0.0]]]),
        ),
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
            id="circle_flat_roll",
            mesh_data=MeshData(
                dimension=2,
                mesh_type=scenarios.m_circle,
                scale=[1],
                mesh_density=[mesh_density],
            ),
            body_prop=scenarios.default_body_prop,
            obstacle_prop=scenarios.default_obstacle_prop,
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

    for scenario in all_scenarios:
        plotter_mapper.print_one_dynamic(
            Solver.solve,
            scenario,
            SettingRandomized.get_setting,
            catalog="EXAMPLES ROLL",
            simulate_dirty_data=False,
            plot_base=False,
            plot_detailed=True,
            plot_animation=plot_animation
        )


if __name__ == "__main__":
    main()
