import deep_conmech.scenarios as scenarios
from conmech.helpers import cmh
from deep_conmech.common import simulation_runner
from deep_conmech.graph.setting.setting_randomized import SettingRandomized
from deep_conmech.scenarios import *
from deep_conmech.graph.model import GraphModelDynamic

def main():
    config = TrainingConfig()
    simulation_runner.run_examples(
        all_scenarios=scenarios.all_print(config),
        file=__file__,
        simulate_dirty_data=True,
        plot_animation=True,
        config=config,
        get_setting_function=GraphModelDynamic.get_setting_function,
    )


if __name__ == "__main__":
    main()
