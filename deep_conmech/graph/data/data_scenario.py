from deep_conmech.common import config
from deep_conmech.graph.data.data_base import *
from deep_conmech.graph.helpers import thh
from deep_conmech.graph.setting.setting_input import SettingInput
from deep_conmech.simulator.calculator import Calculator
from deep_conmech.simulator.setting.setting_forces import *


class ScenariosDatasetDynamic(BaseDatasetDynamic):
    def __init__(
        self,
        base_scenarios,
        episode_steps,
        solve_function,
        relative_path,
        num_workers,
        repetitions=1
    ):
        dim = base_scenarios[0].dim  # TODO: Check other
        self.all_scenarios = base_scenarios * repetitions
        self.solve_function = solve_function
        self.repetitions = repetitions
        self.episode_steps = episode_steps

        data_count = self.episode_steps * len(
            self.all_scenarios
        )  # 10  # config.EPISODE_STEPS

        super().__init__(
            dim,
            relative_path=relative_path,
            data_count=data_count,
            randomize_at_load=True,
            num_workers=num_workers
        )
        self.initialize_data()

    def generate_data_process(self, num_workers, process_id):
        assigned_scenarios = get_assigned_scenarios(
            self.all_scenarios, num_workers, process_id
        )
        data_count = self.episode_steps * len(assigned_scenarios)

        start_index = process_id * data_count
        stop_index = start_index + data_count
        assigned_data_range = range(start_index, stop_index)

        indices_to_do = get_and_check_indices_to_do(
            assigned_data_range, self.path, process_id
        )
        if not indices_to_do:
            return True

        return self.generate_scenario_data(assigned_scenarios, start_index, process_id)

    def generate_scenario_data(
        self, assigned_scenarios, start_index=0, process_id=1,
    ):
        current_index = start_index

        data_count = len(assigned_scenarios) * self.episode_steps
        tqdm_description = f"Process {process_id}"
        step_tqdm = thh.get_tqdm(range(data_count), tqdm_description, process_id)
        for index in step_tqdm:
            ts = (index % self.episode_steps) + 1
            current_time = ts * config.TIMESTEP
            if ts == 1:
                scenario = assigned_scenarios[int(index/self.episode_steps)]
                setting = scenario.get_setting()

                tqdm_description = f"Process {process_id}: Generating {self.relative_path} {scenario.id} data"
                step_tqdm.set_description(tqdm_description)
            if is_memory_overflow(step_tqdm, tqdm_description):
                return False
            
            forces = setting.get_forces_by_function(
                scenario.forces_function, current_time
            )
            setting.prepare(forces)

            a, normalized_a = self.solve_function(setting)
            exact_normalized_a_torch = thh.to_torch_double(normalized_a)

            self.save(setting, exact_normalized_a_torch, current_index)
            self.check_and_print(
                data_count,
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


class TrainingScenariosDatasetDynamic(ScenariosDatasetDynamic):
    def __init__(self, base_scenarios, solve_function, repetitions=1):
        episode_steps = config.EPISODE_STEPS
        num_workers = 1  # config.GENERATION_WORKERS
        super().__init__(
            base_scenarios,
            episode_steps,
            solve_function,
            relative_path="training_scenarios",
            num_workers=num_workers,
            repetitions=repetitions,
        )

    def update_data(self):
        self.clear_and_initialize_data()


class ValidationScenarioDatasetDynamic(ScenariosDatasetDynamic):
    def __init__(self, scenario):
        num_workers = 1 #config.GENERATION_WORKERS
        super().__init__(
            base_scenarios=[scenario],
            episode_steps=config.EPISODE_STEPS,
            solve_function=Calculator.solve_all,
            relative_path=f"validation/{scenario.id}",
            repetitions=1,
            num_workers=num_workers
        )
        