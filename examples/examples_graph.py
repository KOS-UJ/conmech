import deep_conmech.scenarios as scenarios
from conmech.helpers import cmh
from deep_conmech.common.plotter import plotter_mapper
from deep_conmech.graph.setting.setting_randomized import SettingRandomized
from deep_conmech.scenarios import *
from deep_conmech.simulator.solver import Solver


def main(mesh_density=5, final_time=5.0, plot_animation=True):
    for scenario in scenarios.all_print(mesh_density=mesh_density, final_time=final_time):
        plotter_mapper.print_one_dynamic(
            Solver.solve,
            scenario,
            SettingRandomized.get_setting,
            catalog="EXAMPLES GRAPH",
            simulate_dirty_data=True,
            plot_base=False,
            plot_detailed=True,
            plot_animation=plot_animation
        )


if __name__ == "__main__":
    main()
