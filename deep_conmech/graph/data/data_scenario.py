from argparse import ArgumentError
from turtle import up
from typing import List

from deep_conmech.common import config
from deep_conmech.graph.data.data_base import *
from deep_conmech.graph.helpers import thh
from deep_conmech.graph.setting.setting_input import SettingInput
from deep_conmech.scenarios import Scenario
from deep_conmech.simulator.calculator import Calculator
from deep_conmech.simulator.setting.setting_forces import *


class ScenariosDatasetDynamic(BaseDatasetDynamic):
    def __init__(
        self,
        base_scenarios: List[Scenario],
        solve_function,
        relative_path,
        num_workers,
        repetitions=1,
    ):
        dimensions = set([s.mesh_data.dimension for s in base_scenarios])
        if len(dimensions) != 1:
            raise ArgumentError("Incorrect data")
        dimension = dimensions.pop()
        self.all_scenarios = base_scenarios * repetitions
        self.solve_function = solve_function
        self.repetitions = repetitions

        data_count = np.sum([s.time_data.episode_steps for s in self.all_scenarios])

        super().__init__(
            dimension,
            relative_path=relative_path,
            data_count=data_count,
            randomize_at_load=True,
            num_workers=num_workers,
        )
        self.initialize_data()

    def generate_data_process(self, num_workers, process_id):
        assigned_scenarios = get_assigned_scenarios(
            self.all_scenarios, num_workers, process_id
        )

        start_index = process_id * self.data_count
        stop_index = start_index + self.data_count
        assigned_data_range = range(start_index, stop_index)

        indices_to_do = get_and_check_indices_to_do(
            assigned_data_range, self.path, process_id
        )
        if not indices_to_do:
            return True

        return self.generate_scenario_data(assigned_scenarios, start_index, process_id)

    def generate_scenario_data(
        self, assigned_scenarios: List[Scenario], start_index=0, process_id=1,
    ):
        current_index = start_index
        tqdm_description = f"Process {process_id}"
        step_tqdm = cmh.get_tqdm(range(self.data_count), tqdm_description, process_id)
        scenario = assigned_scenarios[0]
        for index in step_tqdm:
            ts = (index % scenario.time_data.episode_steps) + 1
            if ts == 1:
                scenario = assigned_scenarios[
                    int(index / scenario.time_data.episode_steps)
                ]
                setting = self.get_setting_input(scenario)

                tqdm_description = f"Process {process_id}: Generating {self.relative_path} {scenario.id} data"
                step_tqdm.set_description(tqdm_description)
            if is_memory_overflow(step_tqdm, tqdm_description):
                return False

            current_time = ts * setting.time_step
            forces = setting.get_forces_by_function(
                scenario.forces_function, current_time
            )
            setting.prepare(forces)

            a, normalized_a = self.solve_function(setting)
            exact_normalized_a_torch = thh.to_torch_double(normalized_a)

            self.save(setting, exact_normalized_a_torch, current_index)
            self.check_and_print(
                self.data_count, current_index, setting, step_tqdm, tqdm_description,
            )

            # setting = setting.get_copy()
            setting.iterate_self(a)
            current_index += 1

        step_tqdm.set_description(f"{step_tqdm.desc} - done")
        return True


class TrainingScenariosDatasetDynamic(ScenariosDatasetDynamic):
    def __init__(
        self, base_scenarios, solve_function, perform_data_update=False, repetitions=1
    ):
        self.perform_data_update = perform_data_update
        super().__init__(
            base_scenarios,
            solve_function,
            relative_path="training_scenarios",
            num_workers=config.GENERATION_WORKERS,
            repetitions=repetitions,
        )

    def update_data(self):
        if self.perform_data_update:
            self.clear_and_initialize_data()


class ValidationScenarioDatasetDynamic(ScenariosDatasetDynamic):
    def __init__(self, scenario):
        super().__init__(
            base_scenarios=[scenario],
            solve_function=Calculator.solve_all,
            relative_path=f"validation/{scenario.id}",
            repetitions=1,
            num_workers=config.GENERATION_WORKERS,
        )

