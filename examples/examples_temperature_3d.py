import deep_conmech.scenarios as scenarios
from conmech.helpers import cmh
from deep_conmech.common.plotter import plotter_mapper
from deep_conmech.scenarios import *
from deep_conmech.simulator.setting.setting_temperature import \
    SettingTemperature
from deep_conmech.simulator.solver import Solver


def main(mesh_density=3, final_time=3, plot_animation=True):
    all_temp_body_prop = [
        get_temp_body_prop(
            C_coeff=np.array([[1, 0], [0, 0.5]]),
            K_coeff=np.array([[0.1, 0], [0, 0.1]]),
        ),
    ]

    all_scenarios = []
    all_scenarios.extend(
        [
            TemperatureScenario(
                id="temperature_3d_cube_throw",
                mesh_data=MeshData(
                    dimension=3,
                    mesh_type=m_cube_3d,
                    scale=[1],
                    mesh_density=[mesh_density],
                ),
                body_prop=default_temp_body_prop,
                obstacle_prop=default_obstacle_prop,
                schedule=Schedule(final_time=final_time),
                forces_function=f_rotate_3d,
                obstacles=np.array([[[-1.0, 0.0, 1.0]], [[2.0, 0.0, 0.0]]]),
                heat_function=np.array([0]),
            ),
        ]
    )
    
    for scenario in all_scenarios:
        print("-----")
        plotter_mapper.print_one_dynamic(
            Solver.solve_with_temperature,
            scenario,
            SettingTemperature.get_setting,
            catalog="EXAMPLES TEMPERATURE 3D",
            simulate_dirty_data=False,
            plot_base=False,
            plot_detailed=True,
            plot_animation=plot_animation
        )


if __name__ == "__main__":
    main()
