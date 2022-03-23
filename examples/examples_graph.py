from deep_conmech.graph.setting.setting_randomized import SettingRandomized
import deep_conmech.scenarios as scenarios
from conmech.helpers import cmh
from deep_conmech.common.plotter import plotter_mapper
from deep_conmech.scenarios import *
from deep_conmech.simulator.calculator import Calculator


def main():
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


if __name__ == "__main__":
    main()
