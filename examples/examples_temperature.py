import deep_conmech.scenarios as scenarios
from conmech.helpers import cmh
from deep_conmech.common.plotter import plotter_mapper
from deep_conmech.scenarios import *
from deep_conmech.simulator.calculator import Calculator
from deep_conmech.simulator.setting.setting_temperature import SettingTemperature


def main(mesh_density=3, final_time=3.0):
    all_scenarios = [
        Scenario(
            "polygon_rotate",
            MeshData(
                dimension=2,
                mesh_type=m_polygon,
                scale=[1],
                mesh_density=[mesh_density],
                is_adaptive=False,
            ),
            body_prop,
            obstacle_prop,
            schedule=Schedule(final_time=final_time),
            forces_function=np.array([1, 0]),  # f_rotate,
            obstacles=o_front,
        ),
        Scenario(
            id="cube_throw",
            mesh_data=MeshData(
                dimension=3, mesh_type=m_cube_3d, scale=[1], mesh_density=[mesh_density]
            ),
            body_prop=body_prop,
            obstacle_prop=obstacle_prop,
            schedule=Schedule(final_time=final_time),
            forces_function=f_rotate_3d,
            obstacles=np.array([[[-1.0, 0.0, 1.0]], [[2.0, 0.0, 0.0]]]),
        ),
    ]
    for scenario in all_scenarios:
        plotter_mapper.print_one_dynamic(
            Calculator.solve_with_temperature,
            scenario,
            SettingTemperature.get_setting,
            catalog="EXAMPLES TEMPERATURE",
            simulate_dirty_data=False,
            draw_base=False,
            draw_detailed=True,
        )


if __name__ == "__main__":
    main()
