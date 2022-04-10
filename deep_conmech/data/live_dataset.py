from typing import List

from conmech.scenarios.scenarios import Scenario
from deep_conmech.data.scenario_dataset import ScenariosDataset
from deep_conmech.graph.net import CustomGraphNet
from deep_conmech.training_config import TrainingConfig


class LiveDataset(ScenariosDataset):
    def __init__(
        self,
        description: str,
        all_scenarios: List[Scenario],
        net: CustomGraphNet,
        load_to_ram: bool,
        config: TrainingConfig,
    ):
        self.initial_final_times = [scenario.schedule.final_time for scenario in all_scenarios]
        for scenario in all_scenarios:
            scenario.schedule.final_time = 0.1

        super().__init__(
            description=f"{description}_live",
            all_scenarios=all_scenarios,
            solve_function=net.solve_all,
            load_to_ram=load_to_ram,
            config=config,
        )

    def update_data(self):
        super().update_data()
        for i, scenario in enumerate(self.all_scenarios):
            scenario.schedule.final_time = min(
                scenario.schedule.final_time + 0.1, self.initial_final_times[i]
            )
        print(
            f"FIRST SCENARIO FINAL TIME {self.all_scenarios[0].schedule.final_time}, SET VERSION {self.set_version}, DATA COUNT {self.data_count}"
        )
        # cmh.clear_folder(self.main_directory)
        # cmh.clear_folder(self.images_directory)
        self.initialize_data()
        self.set_version += 1
