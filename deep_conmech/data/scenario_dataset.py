from ctypes import ArgumentError
from typing import Callable, List

import numpy as np

from conmech.helpers import cmh, pkh
from conmech.scenarios.scenarios import Scenario
from deep_conmech.data.base_dataset import (
    BaseDataset,
    get_assigned_scenarios,
    is_memory_overflow,
)
from deep_conmech.helpers import thh
from deep_conmech.training_config import TrainingConfig


class ScenariosDataset(BaseDataset):
    def __init__(
        self,
        description: str,
        all_scenarios: List[Scenario],
        skip_index: int,
        solve_function: Callable,
        load_to_ram: bool,
        config: TrainingConfig,
    ):
        self.all_scenarios = all_scenarios
        self.skip_index = skip_index
        self.solve_function = solve_function

        super().__init__(
            description=description,
            dimension=self.check_and_get_dimension(all_scenarios),
            data_count=self.get_data_count(self.all_scenarios),
            randomize_at_load=True,
            num_workers=1,  # TODO: #65 Check
            load_to_ram=load_to_ram,
            config=config,
        )
        self.initialize_data()

    def check_and_get_dimension(self, scenarios):
        dimensions = set([s.mesh_prop.dimension for s in scenarios])
        if len(dimensions) != 1:
            raise ArgumentError("Incorrect data")
        return dimensions.pop()

    def get_data_count(self, scenarios):
        return np.sum([int(s.schedule.episode_steps / self.skip_index) for s in scenarios])

    @property
    def data_size_id(self):
        return f"f:{self.config.td.FINAL_TIME}_i:{self.skip_index}"

    def generate_data_process(self, num_workers, process_id):
        assigned_scenarios = get_assigned_scenarios(self.all_scenarios, num_workers, process_id)
        assigned_data_count = np.sum([s.schedule.episode_steps for s in self.all_scenarios])
        start_index = process_id * assigned_data_count
        current_index = start_index
        tqdm_description = f"P{process_id}: Generating {self.description}"
        step_tqdm = cmh.get_tqdm(
            range(assigned_data_count),
            config=self.config,
            desc=tqdm_description,
            position=process_id,
        )
        scenario = assigned_scenarios[0]

        settings_file, file_meta = pkh.open_files_append_pickle(self.data_path)
        with settings_file, file_meta:
            for index in step_tqdm:
                episode_steps = scenario.schedule.episode_steps
                ts = (index % episode_steps) + 1
                if ts == 1:
                    scenario = assigned_scenarios[int(index / episode_steps)]
                    setting = self.get_scene_input(scenario=scenario, config=self.config)

                if is_memory_overflow(
                    config=self.config,
                    step_tqdm=step_tqdm,
                    tqdm_description=tqdm_description,
                ):
                    return False

                current_time = ts * setting.time_step
                forces = scenario.get_forces_by_function(setting, current_time)
                setting.prepare(forces)

                a, normalized_a = self.solve_function(setting)
                exact_normalized_a_torch = thh.to_torch_double(normalized_a)

                if index % self.skip_index == 0:
                    pkh.append_pickle(
                        setting=setting, settings_file=settings_file, file_meta=file_meta
                    )  # exact_normalized_a_torch

                self.check_and_print(
                    self.data_count,
                    current_index,
                    setting,
                    step_tqdm,
                    tqdm_description,
                )

                # setting = setting.get_copy()
                setting.iterate_self(a)
                current_index += 1

        step_tqdm.set_description(f"{step_tqdm.desc} - done")
        return True
