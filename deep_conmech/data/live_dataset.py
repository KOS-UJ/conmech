from ctypes import ArgumentError
from typing import Callable, List

import numpy as np

from conmech.helpers import cmh, pkh
from conmech.solvers.calculator import Calculator
from deep_conmech.data.scenario_dataset import ScenariosDataset
from deep_conmech.training_config import TrainingConfig
from deep_conmech.data.base_dataset import BaseDataset, get_assigned_scenarios, \
    is_memory_overflow
from deep_conmech.helpers import thh
from conmech.scenarios.scenarios import Scenario
from conmech.scenarios import scenarios
from deep_conmech.graph.net import CustomGraphNet



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
            scenario.schedule.final_time = 0.2

        super().__init__(
            description=f"{description}_live", all_scenarios=all_scenarios, solve_function=net.solve_all, 
            load_to_ram=load_to_ram, config=config
        )



    def update_data(self):
        super().update_data()
        for i, scenario in enumerate(self.all_scenarios):
            if scenario.schedule.final_time < self.initial_final_times[i]:
                scenario.schedule.final_time += 0.4
        #cmh.clear_folder(self.main_directory)
        #cmh.clear_folder(self.images_directory)
        self.initialize_data()
        self.set_version += 1


