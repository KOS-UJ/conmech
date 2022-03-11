from matplotlib.pyplot import draw

from deep_conmech.graph.helpers import thh
import deep_conmech.common.config as config
import deep_conmech.scenarios as scenarios
from deep_conmech.common.plotter import plotter_mapper
from deep_conmech.simulator.calculator import Calculator


def main():
    for scenario in scenarios.all_simulator:
        path = f"SIMULATOR 2D - {thh.CURRENT_TIME}"

        plotter_mapper.print_one_dynamic(
            Calculator.solve,
            scenario,
            path,
            simulate_dirty_data=config.SIMULATE_DIRTY_DATA,
            print_base=False,
            description="Printing simulator 2D",
        )

if __name__ == "__main__":
    main()
