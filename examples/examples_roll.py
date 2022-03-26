import deep_conmech.scenarios as scenarios
from conmech.helpers import cmh
from deep_conmech.common.plotter import plotter_mapper
from deep_conmech.graph.setting.setting_randomized import SettingRandomized
from deep_conmech.scenarios import *
from deep_conmech.simulator.calculator import Calculator


def main(mesh_density=4, final_time=3.0):
    all_scenarios = [
        Scenario(
            id="rectangle_small",
            mesh_data=MeshData(
                dimension=2,
                mesh_type=scenarios.m_rectangle,
                scale=[1],
                mesh_density=[4],
            ),
            body_prop=scenarios.body_prop,
            obstacle_prop=scenarios.obstacle_prop,
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
            body_prop=scenarios.body_prop,
            obstacle_prop=scenarios.obstacle_prop,
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
            body_prop=scenarios.body_prop,
            obstacle_prop=scenarios.obstacle_prop,
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
            body_prop=scenarios.body_prop,
            obstacle_prop=scenarios.obstacle_prop,
            schedule=Schedule(final_time=final_time),
            forces_function=np.array([2.0, -0.5]),
            obstacles=np.array([[[0.0, 1.0]], [[0.0, 0.0]]]),
        ),
    ]

    for scenario in all_scenarios:
        plotter_mapper.print_one_dynamic(
            Calculator.solve,
            scenario,
            SettingRandomized.get_setting,
            catalog="EXAMPLES ROLL",
            simulate_dirty_data=False,
            draw_base=False,
            draw_detailed=True,
        )


if __name__ == "__main__":
    main()
