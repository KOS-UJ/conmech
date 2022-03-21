from deep_conmech.graph.setting.setting_randomized import SettingRandomized
import deep_conmech.scenarios as scenarios
from conmech.helpers import cmh
from deep_conmech.common.plotter import plotter_mapper
from deep_conmech.scenarios import *
from deep_conmech.simulator.calculator import Calculator


def main():
    path = f"EXAMPLES GRAPH - {cmh.CURRENT_TIME}"

    for scenario in scenarios.all_print:
        plotter_mapper.print_one_dynamic(
            Calculator.solve,
            scenario,
            SettingRandomized.get_setting,
            path,
            simulate_dirty_data=False,
            draw_base=False,
            draw_detailed=True,
            description="Printing",
        )


if __name__ == "__main__":
    main()
