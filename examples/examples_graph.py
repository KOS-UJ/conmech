from deep_conmech.graph.setting.setting_randomized import SettingRandomized
import deep_conmech.scenarios as scenarios
from conmech.helpers import cmh
from deep_conmech.common.plotter import plotter_mapper
from deep_conmech.scenarios import *
from deep_conmech.simulator.calculator import Calculator


def main(mesh_density=4, final_time=3.0):
    for scenario in scenarios.all_print(mesh_density=mesh_density, final_time=final_time):
        plotter_mapper.print_one_dynamic(
            Calculator.solve,
            scenario,
            SettingRandomized.get_setting,
            catalog="EXAMPLES GRAPH",
            simulate_dirty_data=True,
            draw_base=False,
            draw_detailed=True
        )


if __name__ == "__main__":
    main()
