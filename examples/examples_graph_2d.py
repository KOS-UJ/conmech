import deep_conmech.scenarios as scenarios
from conmech.helpers import cmh
from deep_conmech.common import simulation_runner
from deep_conmech.graph.setting.setting_randomized import SettingRandomized
from deep_conmech.scenarios import *
from deep_conmech.simulator.solver import Solver


def main(mesh_density=5, final_time=5, plot_animation=True):
    simulation_runner.run_examples(
        all_scenarios=scenarios.all_print(
            mesh_density=mesh_density, final_time=final_time
        ),
        file=__file__,
        plot_animation=plot_animation,
    )


if __name__ == "__main__":
    main()
