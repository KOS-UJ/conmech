from conmech.helpers import mph
from deep_conmech.common import config
from deep_conmech.graph.data.data_base import *
from deep_conmech.graph.helpers import thh
from deep_conmech.graph.setting.setting_input import SettingInput
from deep_conmech.simulator.calculator import Calculator
from deep_conmech.simulator.setting.setting_forces import *



class TrainingScenariosDatasetDynamic(BaseDatasetDynamic):
    def __init__(self, base_scenarios, repetitions=2):
        dim = base_scenarios[0].dim  # TODO: Check other
        self.all_scenarios = base_scenarios * repetitions
        data_count = config.EPISODE_STEPS * len(self.all_scenarios)
        super().__init__(
            dim,
            relative_path="training_scenarios",
            data_count=data_count,
            randomize_at_load=True,
        )
        self.repetitions = repetitions
        self.initialize_data()

    def generate_all_data(self):
        # randomly choose scenario parameters
        num_workers = config.GENERATION_WORKERS
        # for process_id in range(num_workers):
        #    generate_scenario_data_process(self, self.all_scenarios, num_workers=num_workers, queue=None, process_id=process_id)

        result = False
        while result is False:
            result = mph.run_processes(
                self.generate_scenario_data_process, (self, self.all_scenarios), num_workers,
            )
            if result is False:
                print("Restarting data generation")



class ValidationScenarioDatasetDynamic(BaseDatasetDynamic):
    def __init__(self, scenario):
        super().__init__(
            dim=scenario.dim,
            relative_path=f"validation/{scenario.id}",
            data_count=config.EPISODE_STEPS,
            randomize_at_load=False,
        )
        self.scenario = scenario
        self.initialize_data()

    def generate_all_data(self):
        result = False
        while result is False:
            result = mph.run_processes(
                self.generate_scenario_data_process, (self, [self.scenario]), 1,
            )
