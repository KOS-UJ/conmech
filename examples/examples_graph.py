from deep_conmech.simulator.setting.setting_temperature import SettingTemperature
import deep_conmech.scenarios as scenarios
from conmech.helpers import cmh
from deep_conmech.common.plotter import plotter_mapper
from deep_conmech.scenarios import *
from deep_conmech.simulator.calculator import Calculator


def main():

    scenario = Scenario(
        "polygon_rotate",
        MeshData(
            dimension=2,
            mesh_type=m_polygon,
            scale=[1],
            mesh_density=[3],
            is_adaptive=False,
        ),
        body_prop,
        obstacle_prop,
        schedule=Schedule(final_time=5.0),
        forces_function=f_rotate,
        obstacles=o_front,
    )
    plotter_mapper.print_one_dynamic(
        Calculator.solve_with_temperature,
        scenario,
        SettingTemperature.get_setting,
        catalog="EXAMPLES GRAPH",
        simulate_dirty_data=True, ###
        draw_base=False,
        draw_detailed=True,
        description="Examples graph",
        with_temperatue=True
    )

    '''
    for scenario in scenarios.all_print:
        plotter_mapper.print_one_dynamic(
            Calculator.solve,
            scenario,
            SettingRandomized.get_setting,
            catalog="EXAMPLES GRAPH",
            simulate_dirty_data=False,
            draw_base=False,
            draw_detailed=True,
            description="Examples graph",
        )
    '''


if __name__ == "__main__":
    main()
