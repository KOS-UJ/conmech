from matplotlib.pyplot import draw

from deep_conmech.graph.helpers import thh
import deep_conmech.common.config as config
import deep_conmech.common.scenarios as scenarios
import deep_conmech.common.plotter.plotter_mapper as mapper
from deep_conmech.simulator.calculator import Calculator


def run_conmech_dynamic(all_scenatrios):

    for scenario in all_scenatrios:
        path = f"SIMULATOR 2D - {thh.CURRENT_TIME}"

        mapper.print_one_dynamic(
            Calculator.solve,
            scenario,
            path,
            simulate_dirty_data=config.SIMULATE_DIRTY_DATA_SIMULATOR,
            print_base=False,
            description="Printing simulator",
        )


def main():
    run_conmech_dynamic(scenarios.all_simulator)


if __name__ == "__main__":
    main()
