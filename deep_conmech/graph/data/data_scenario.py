from argparse import ArgumentError
from typing import List

from deep_conmech.graph.data.data_base import *
from deep_conmech.graph.helpers import thh
from deep_conmech.scenarios import Scenario
from deep_conmech.simulator.setting.setting_forces import *
from deep_conmech.simulator.solver import Solver


class ScenariosDatasetDynamic(BaseDatasetDynamic):
    def __init__(
            self,
            all_scenarios: List[Scenario],
            solve_function,
            relative_path,
            num_workers,
            config: Config,
    ):
        self.all_scenarios = all_scenarios
        self.solve_function = solve_function

        super().__init__(
            self.check_and_get_dimension(all_scenarios),
            relative_path=relative_path,
            data_count=self.get_data_count(self.all_scenarios),
            randomize_at_load=True,
            num_workers=num_workers,
            config=config,
        )
        self.initialize_data()

    def check_and_get_dimension(self, scenarios):
        dimensions = set([s.mesh_data.dimension for s in scenarios])
        if len(dimensions) != 1:
            raise ArgumentError("Incorrect data")
        return dimensions.pop()

    def get_data_count(self, scenarios):
        return np.sum([s.schedule.episode_steps for s in scenarios])

    def generate_data_process(self, num_workers, process_id):
        assigned_scenarios = get_assigned_scenarios(
            self.all_scenarios, num_workers, process_id
        )
        # TODO: Will not work for nonequal assigned_data_counts
        assigned_data_count = self.get_data_count(assigned_scenarios)

        start_index = process_id * assigned_data_count
        stop_index = start_index + assigned_data_count
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
        assigned_data_count = self.get_data_count(assigned_scenarios)
        step_tqdm = cmh.get_tqdm(
            range(assigned_data_count),
            config=self.config,
            desc=tqdm_description,
            position=process_id,
        )
        scenario = assigned_scenarios[0]
        for index in step_tqdm:
            episode_steps = scenario.schedule.episode_steps
            ts = (index % episode_steps) + 1
            if ts == 1:
                scenario = assigned_scenarios[int(index / episode_steps)]
                setting = self.get_setting_input(scenario=scenario, config=self.config)

                tqdm_description = f"Process {process_id}: Generating {self.relative_path} {scenario.id} data"
                step_tqdm.set_description(tqdm_description)
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
            self, all_scenarios, solve_function, config: Config, perform_data_update=False
    ):
        self.perform_data_update = perform_data_update
        super().__init__(
            all_scenarios=all_scenarios,
            solve_function=solve_function,
            relative_path="training_scenarios",
            num_workers=1,  # config.GENERATION_WORKERS,
            config=config,
        )

    def update_data(self):
        if self.perform_data_update:
            self.clear_and_initialize_data()


class ValidationScenarioDatasetDynamic(ScenariosDatasetDynamic):
    def __init__(self, all_scenarios, id, config: Config):
        super().__init__(
            all_scenarios=all_scenarios,
            solve_function=Solver.solve_all,
            relative_path=f"validation/{id}",
            num_workers=1,  ###
            config=config,
        )
